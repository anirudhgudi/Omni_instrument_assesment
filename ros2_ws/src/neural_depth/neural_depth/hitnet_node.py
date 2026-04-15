import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime
import tf2_ros


class HitnetNode(Node):
    def __init__(self):
        super().__init__('hitnet_depth_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)

        model_path = self.get_parameter('model_path').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value

        self.get_logger().info(f"Loading HITNET ONNX from {model_path}...")

        # This HITNet export is small enough for CPU. Keeping CPU-only avoids
        # CUDA provider library mismatches across TensorRT base images.
        providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.get_logger().info(f"HITNET Input Shape: {self.input_shape}")

        self.model_outputs = self.session.get_outputs()
        self.output_names = [self.model_outputs[i].name for i in range(len(self.model_outputs))]

        self.bridge = CvBridge()
        self.baseline = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Output publishers
        self.depth_pub = self.create_publisher(Image, '/omni_depth/depth_registered', pub_qos)
        self.cam_info_pub = self.create_publisher(CameraInfo, '/omni_depth/camera_info', pub_qos)
        # Colorized depth image for easy RViz viewing
        self.depth_color_pub = self.create_publisher(Image, '/omni_depth/depth_colorized', pub_qos)

        self.left_sub = Subscriber(self, Image, '/zed/zedxm/left/color/rect/image', qos_profile=sub_qos)
        self.right_sub = Subscriber(self, Image, '/zed/zedxm/right/color/rect/image', qos_profile=sub_qos)
        self.left_cam_info_sub = Subscriber(self, CameraInfo, '/zed/zedxm/left/color/rect/image/camera_info', qos_profile=sub_qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub, self.left_cam_info_sub],
            queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.image_callback)
        self.get_logger().info("Hitnet Depth Node Initialized!")

    def _get_baseline(self) -> float | None:
        if self.baseline is not None:
            return self.baseline
        try:
            t = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame_optical',
                'zed_right_camera_frame_optical',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            self.baseline = abs(t.transform.translation.x)
            self.get_logger().info(f'Baseline from TF: {self.baseline:.6f} m')
            return self.baseline
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return None

    def prepare_input(self, left_image, right_image):
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

        left_gray = cv2.resize(left_gray, (self.input_width, self.input_height), cv2.INTER_AREA)
        right_gray = cv2.resize(right_gray, (self.input_width, self.input_height), cv2.INTER_AREA)

        stereo = np.stack([left_gray, right_gray], axis=0).astype(np.float32) / 255.0
        return np.expand_dims(stereo, axis=0)

    def image_callback(self, left_msg, right_msg, l_info):
        baseline = self._get_baseline()
        if baseline is None:
            return

        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, "rgb8")
            right_img = self.bridge.imgmsg_to_cv2(right_msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        orig_height, orig_width = left_img.shape[:2]

        stereo_tensor = self.prepare_input(left_img, right_img)

        fx = l_info.k[0]
        if fx <= 0.0:
            self.get_logger().warn('Invalid focal length in CameraInfo', throttle_duration_sec=5.0)
            return

        # Inference
        disparity_map = self.session.run(self.output_names, {
            self.input_names[0]: stereo_tensor,
        })[0]

        # disparity_map shape is (1, 1, H, W). Remove batch and channel
        disparity_map = np.squeeze(disparity_map)

        # Resize back to original image size
        # Disparity needs to be scaled by the resizing factor to maintain physical correctness
        width_ratio = orig_width / self.input_width
        disparity_map = cv2.resize(disparity_map, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        disparity_map = disparity_map * width_ratio

        # Convert disparity to depth
        # Z = (f * B) / d
        depth_map = np.zeros_like(disparity_map, dtype=np.float32)
        valid_disp = disparity_map > 0.5
        depth_map[valid_disp] = (fx * baseline) / disparity_map[valid_disp]

        depth_map[depth_map < self.min_depth] = 0.0
        depth_map[depth_map > self.max_depth] = 0.0

        # Publish depth (32FC1)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
        depth_msg.header = left_msg.header
        self.depth_pub.publish(depth_msg)

        # Publish camera_info (copy from left cam, same intrinsics apply to depth)
        cam_info_msg = CameraInfo()
        cam_info_msg.header = left_msg.header
        cam_info_msg.height = l_info.height
        cam_info_msg.width = l_info.width
        cam_info_msg.distortion_model = l_info.distortion_model
        cam_info_msg.d = l_info.d
        cam_info_msg.k = l_info.k
        cam_info_msg.r = l_info.r
        cam_info_msg.p = l_info.p
        self.cam_info_pub.publish(cam_info_msg)

        # Publish colorized depth for RViz Image panel
        normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        colorized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        color_msg = self.bridge.cv2_to_imgmsg(colorized, encoding="bgr8")
        color_msg.header = left_msg.header
        self.depth_color_pub.publish(color_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HitnetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
