"""Exercise 1: display the camera feed in an OpenCV window."""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraViewer(Node):
    """Subscribe to the robot camera and display the live feed."""

    def __init__(self) -> None:
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10,
        )

    def callback(self, data: Image) -> None:
        """Convert ROS images to OpenCV and show them."""
        try:
            image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.resizeWindow('camera_feed', 640, 480)
        cv2.waitKey(1)


def main(args=None) -> None:
    """Run the camera viewer node until interrupted."""
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
