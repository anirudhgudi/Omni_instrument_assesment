"""
Stereo depth estimation using OpenCV StereoSGBM + WLS post-filtering.

Subscribes to rectified left/right gray stereo pairs and camera info,
computes disparity via StereoSGBM, refines with ximgproc WLS filter,
and publishes a metric depth image (32FC1, meters) for the TSDF pipeline.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import message_filters
import tf2_ros


class StereoDepthNode(Node):
    """ROS 2 node: StereoSGBM + WLS disparity → metric depth."""

    def __init__(self):
        super().__init__('stereo_depth_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('num_disparities', 128)
        self.declare_parameter('block_size', 5)
        self.declare_parameter('wls_lambda', 8000.0)
        self.declare_parameter('wls_sigma', 1.5)

        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        num_disparities = self.get_parameter('num_disparities').value
        block_size = self.get_parameter('block_size').value
        wls_lambda = self.get_parameter('wls_lambda').value
        wls_sigma = self.get_parameter('wls_sigma').value

        # ── OpenCV StereoSGBM setup ─────────────────────────────────
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3P,
        )

        # Right matcher for WLS filter
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # WLS filter
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.left_matcher
        )
        self.wls_filter.setLambda(wls_lambda)
        self.wls_filter.setSigmaColor(wls_sigma)

        # ── Bridge & state ──────────────────────────────────────────
        self.bridge = CvBridge()
        self.baseline = None  # Will be extracted from TF

        # ── TF2 for baseline extraction ─────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── QoS profiles ────────────────────────────────────────────
        # Subscribers match bag BEST_EFFORT sensor data
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Publishers use RELIABLE so composable nodes (depth_image_proc)
        # can subscribe — image_transport::CameraSubscriber defaults to RELIABLE
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ──────────────────────────────────────────────
        self.depth_pub = self.create_publisher(
            Image,
            '/omni_depth/depth_registered',
            pub_qos,
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo,
            # image_transport::CameraSubscriber derives camera_info by taking
            # the image topic's parent namespace + /camera_info, so:
            #   /omni_depth/depth_registered  →  /omni_depth/camera_info
            '/omni_depth/camera_info',
            pub_qos,
        )
        # Colorized depth image for RViz visualization
        self.depth_color_pub = self.create_publisher(
            Image,
            '/omni_depth/depth_colorized',
            pub_qos,
        )

        # ── Synchronised subscribers ────────────────────────────────
        self.sub_left = message_filters.Subscriber(
            self,
            Image,
            '/zed/zedxm/left/gray/rect/image',
            qos_profile=sub_qos,
        )
        self.sub_right = message_filters.Subscriber(
            self,
            Image,
            '/zed/zedxm/right/gray/rect/image',
            qos_profile=sub_qos,
        )
        self.sub_info = message_filters.Subscriber(
            self,
            CameraInfo,
            '/zed/zedxm/left/gray/rect/image/camera_info',
            qos_profile=sub_qos,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right, self.sub_info],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self.stereo_callback)

        self.get_logger().info(
            f'StereoSGBM+WLS node started  '
            f'(numDisp={num_disparities}, block={block_size}, '
            f'λ={wls_lambda}, σ={wls_sigma})'
        )

    # ────────────────────────────────────────────────────────────────
    #  Baseline from TF
    # ────────────────────────────────────────────────────────────────
    def _get_baseline(self) -> float | None:
        """Extract baseline (meters) from TF between left and right optical frames."""
        if self.baseline is not None:
            return self.baseline

        try:
            t = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame_optical',
                'zed_right_camera_frame_optical',
                rclpy.time.Time(),          # latest available
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            # Baseline is the absolute x-translation between the two frames
            self.baseline = abs(t.transform.translation.x)
            self.get_logger().info(f'Baseline extracted from TF: {self.baseline:.6f} m')
            return self.baseline
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return None

    # ────────────────────────────────────────────────────────────────
    #  Main callback
    # ────────────────────────────────────────────────────────────────
    def stereo_callback(self, left_msg: Image, right_msg: Image, info_msg: CameraInfo):
        """Process a synchronised stereo pair → publish metric depth."""

        baseline = self._get_baseline()
        if baseline is None:
            return

        # Focal length from camera intrinsics (fx)
        fx = info_msg.k[0]
        if fx <= 0.0:
            self.get_logger().warn('Invalid focal length in CameraInfo', throttle_duration_sec=5.0)
            return

        # ── Convert ROS images → OpenCV grayscale ───────────────────
        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')

        # ── Compute disparity (left & right for WLS) ────────────────
        left_disp = self.left_matcher.compute(left_img, right_img)
        right_disp = self.right_matcher.compute(right_img, left_img)

        # ── WLS post-filter ─────────────────────────────────────────
        filtered_disp = self.wls_filter.filter(
            left_disp, left_img, disparity_map_right=right_disp
        )

        # Convert from fixed-point (16-bit, 4 fractional bits) to float
        disparity = filtered_disp.astype(np.float32) / 16.0

        # ── Disparity → Metric depth ───────────────────────────────
        # depth = (fx * baseline) / disparity
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.0
        depth[valid] = (fx * baseline) / disparity[valid]

        # Clamp to valid range
        depth[depth < self.min_depth] = 0.0
        depth[depth > self.max_depth] = 0.0

        # ── Publish depth image ─────────────────────────────────────
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = left_msg.header  # preserve timestamp & frame

        self.depth_pub.publish(depth_msg)

        # ── Publish camera info (same intrinsics as left camera) ────
        depth_info = CameraInfo()
        depth_info.header = left_msg.header
        depth_info.height = info_msg.height
        depth_info.width = info_msg.width
        depth_info.distortion_model = info_msg.distortion_model
        depth_info.d = info_msg.d
        depth_info.k = info_msg.k
        depth_info.r = info_msg.r
        depth_info.p = info_msg.p

        self.depth_info_pub.publish(depth_info)

        # ── Publish colorized depth for RViz Image panel ────────────
        normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        colorized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        color_msg = self.bridge.cv2_to_imgmsg(colorized, encoding='bgr8')
        color_msg.header = left_msg.header
        self.depth_color_pub.publish(color_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
