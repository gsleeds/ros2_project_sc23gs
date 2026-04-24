"""Exercise 3: detect a large green object and report it."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class ColourIdentifier(Node):
    """Detect a green object and log when it is confidently visible."""

    def __init__(self) -> None:
        super().__init__('green_detector')
        self.bridge = CvBridge()
        self.sensitivity = 10
        self.area_threshold = 1500.0
        self.green_detected = False
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10,
        )

    def callback(self, data: Image) -> None:
        """Track the largest green contour in the current frame."""
        try:
            image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        filtered_image = cv2.bitwise_and(image, image, mask=green_mask)
        contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detected_this_frame = False
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            if area > self.area_threshold:
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    center_y = int(moments['m01'] / moments['m00'])
                    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
                    cv2.circle(
                        image,
                        (int(circle_x), int(circle_y)),
                        int(radius),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                    detected_this_frame = True

        if detected_this_frame and not self.green_detected:
            self.get_logger().info('Green object detected')
        elif not detected_this_frame and self.green_detected:
            self.get_logger().info('Green object lost')

        self.green_detected = detected_this_frame

        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('green_mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('green_filtered', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.imshow('green_mask', green_mask)
        cv2.imshow('green_filtered', filtered_image)
        cv2.resizeWindow('camera_feed', 640, 480)
        cv2.resizeWindow('green_mask', 640, 480)
        cv2.resizeWindow('green_filtered', 640, 480)
        cv2.waitKey(1)


def main(args=None) -> None:
    """Run the green detector until interrupted."""
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
