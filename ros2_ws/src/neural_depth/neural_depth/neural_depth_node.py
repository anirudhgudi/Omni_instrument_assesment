"""
Neural stereo depth estimation using RAFT-Stereo.

Subscribes to rectified left/right color stereo pairs and camera info,
runs RAFT-Stereo inference on GPU, converts disparity to metric depth,
and publishes the result for the TSDF pipeline.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import message_filters
import tf2_ros


# ────────────────────────────────────────────────────────────────────
#  Minimal InputPadder (from RAFT-Stereo core/utils/utils.py)
#  Inlined here to avoid path-manipulation fragility.
# ────────────────────────────────────────────────────────────────────
class InputPadder:
    """Pad images so that dimensions are divisible by `divis_by`."""

    def __init__(self, dims, divis_by=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                     pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class NeuralDepthNode(Node):
    """ROS 2 node: RAFT-Stereo disparity → metric depth."""

    def __init__(self):
        super().__init__('neural_depth_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('iters', 24)
        self.declare_parameter('mixed_precision', True)
        self.declare_parameter('corr_implementation', 'alt')
        self.declare_parameter('shared_backbone', False)
        self.declare_parameter('corr_levels', 4)
        self.declare_parameter('corr_radius', 4)
        self.declare_parameter('n_downsample', 2)
        self.declare_parameter('context_norm', 'batch')
        self.declare_parameter('slow_fast_gru', False)
        self.declare_parameter('n_gru_layers', 3)
        self.declare_parameter('hidden_dims', [128, 128, 128])
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)

        model_path = self.get_parameter('model_path').value
        self.iters = self.get_parameter('iters').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value

        # ── Load RAFT-Stereo model ──────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        if not model_path or not os.path.isfile(model_path):
            self.get_logger().fatal(f'Model checkpoint not found: {model_path}')
            raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

        # Add RAFT-Stereo to Python path
        raft_root = self._find_raft_root(model_path)
        if raft_root:
            sys.path.insert(0, raft_root)
            self.get_logger().info(f'Added RAFT-Stereo root to path: {raft_root}')

        # Build args namespace for RAFT-Stereo constructor
        args = self._build_args()

        from core.raft_stereo import RAFTStereo

        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = model.module
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(
            f'RAFT-Stereo loaded from {os.path.basename(model_path)} '
            f'(iters={self.iters}, mixed_precision={args.mixed_precision})'
        )

        # ── Bridge & state ──────────────────────────────────────────
        self.bridge = CvBridge()
        self.baseline = None

        # ── TF2 for baseline extraction ─────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── QoS ─────────────────────────────────────────────────────
        # Bag image topics are BEST_EFFORT sensor data. depth_image_proc uses
        # image_transport camera subscriptions, which expect RELIABLE publishers.
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

        # ── Publishers ──────────────────────────────────────────────
        self.depth_pub = self.create_publisher(
            Image,
            '/omni_depth/depth_registered',
            pub_qos,
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo,
            '/omni_depth/camera_info',
            pub_qos,
        )

        # ── Synchronised subscribers (COLOR rectified) ──────────────
        self.sub_left = message_filters.Subscriber(
            self,
            Image,
            '/zed/zedxm/left/color/rect/image',
            qos_profile=sub_qos,
        )
        self.sub_right = message_filters.Subscriber(
            self,
            Image,
            '/zed/zedxm/right/color/rect/image',
            qos_profile=sub_qos,
        )
        self.sub_info = message_filters.Subscriber(
            self,
            CameraInfo,
            '/zed/zedxm/left/color/rect/image/camera_info',
            qos_profile=sub_qos,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_right, self.sub_info],
            queue_size=5,
            slop=0.05,
        )
        self.sync.registerCallback(self.stereo_callback)

        self.frame_count = 0
        self.get_logger().info('Neural depth node (RAFT-Stereo) ready — waiting for images...')

    # ────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────
    def _find_raft_root(self, model_path: str) -> str | None:
        """Walk up from model_path to find the RAFT-Stereo repo root (contains core/)."""
        d = os.path.dirname(os.path.abspath(model_path))
        for _ in range(5):
            if os.path.isdir(os.path.join(d, 'core')):
                return d
            d = os.path.dirname(d)
        return None

    def _build_args(self):
        """Build an argparse-like namespace for RAFTStereo constructor."""
        import argparse
        args = argparse.Namespace()
        args.mixed_precision = self.get_parameter('mixed_precision').value
        args.corr_implementation = self.get_parameter('corr_implementation').value
        args.shared_backbone = self.get_parameter('shared_backbone').value
        args.corr_levels = self.get_parameter('corr_levels').value
        args.corr_radius = self.get_parameter('corr_radius').value
        args.n_downsample = self.get_parameter('n_downsample').value
        args.context_norm = self.get_parameter('context_norm').value
        args.slow_fast_gru = self.get_parameter('slow_fast_gru').value
        args.n_gru_layers = self.get_parameter('n_gru_layers').value
        args.hidden_dims = self.get_parameter('hidden_dims').value
        return args

    def _get_baseline(self) -> float | None:
        """Extract baseline (meters) from TF between left and right optical frames."""
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

    # ────────────────────────────────────────────────────────────────
    #  Main callback
    # ────────────────────────────────────────────────────────────────
    def stereo_callback(self, left_msg: Image, right_msg: Image, info_msg: CameraInfo):
        """Process a synchronised stereo pair → publish metric depth via RAFT-Stereo."""

        baseline = self._get_baseline()
        if baseline is None:
            return

        fx = info_msg.k[0]
        if fx <= 0.0:
            self.get_logger().warn('Invalid focal length in CameraInfo', throttle_duration_sec=5.0)
            return

        # ── Convert ROS images → numpy RGB ──────────────────────────
        left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='rgb8')
        right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='rgb8')

        # ── Numpy → Torch [1, 3, H, W] float32 on GPU ──────────────
        img1 = torch.from_numpy(left_cv).permute(2, 0, 1).float()[None].to(self.device)
        img2 = torch.from_numpy(right_cv).permute(2, 0, 1).float()[None].to(self.device)

        # ── Pad to be divisible by 2^n_downsample * 8 ──────────────
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)

        # ── RAFT-Stereo inference ───────────────────────────────────
        with torch.no_grad():
            _, flow_up = self.model(img1, img2, iters=self.iters, test_mode=True)

        # Unpad and extract disparity (RAFT outputs negative disparity)
        disparity = padder.unpad(flow_up).squeeze()
        disparity = -disparity.cpu().numpy()  # make positive

        # ── Disparity → Metric depth ───────────────────────────────
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0.5  # minimum half-pixel disparity
        depth[valid] = (fx * baseline) / disparity[valid]

        # Clamp to valid range
        depth[depth < self.min_depth] = 0.0
        depth[depth > self.max_depth] = 0.0

        # ── Publish depth image ─────────────────────────────────────
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = left_msg.header

        self.depth_pub.publish(depth_msg)

        # ── Publish camera info ─────────────────────────────────────
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

        # Log periodically
        self.frame_count += 1
        if self.frame_count % 10 == 1:
            valid_pct = np.count_nonzero(depth > 0) / depth.size * 100
            med_depth = np.median(depth[depth > 0]) if np.any(depth > 0) else 0.0
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{valid_pct:.1f}% valid, median depth {med_depth:.2f}m'
            )


def main(args=None):
    rclpy.init(args=args)
    node = NeuralDepthNode()
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
