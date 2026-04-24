"""Exercise 2: detect two colours and filter the rest of the image."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class ColourIdentifier(Node):
    """Display masks for the selected colours and the filtered result."""

    def __init__(self) -> None:
        super().__init__('colour_identifier')
        self.bridge = CvBridge()
        self.sensitivity = 10
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10,
        )

    def callback(self, data: Image) -> None:
        """Highlight green and blue objects while removing other colours."""
        try:
            image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        combined_mask = cv2.bitwise_or(green_mask, blue_mask)
        filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('combined_mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('filtered_image', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.imshow('combined_mask', combined_mask)
        cv2.imshow('filtered_image', filtered_image)
        cv2.resizeWindow('camera_feed', 640, 480)
        cv2.resizeWindow('combined_mask', 640, 480)
        cv2.resizeWindow('filtered_image', 640, 480)
        cv2.waitKey(1)


def main(args=None) -> None:
    """Run the colour filtering node until interrupted."""
    rclpy.init(args=args)
    node = ColourIdentifier()

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
