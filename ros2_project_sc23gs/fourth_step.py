"""Exercise 4: follow green and stop when blue is visible."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


class Robot(Node):
    """Use colour detection to follow green and stop for blue."""

    def __init__(self) -> None:
        super().__init__('colour_following_robot')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.sensitivity = 10
        self.detect_area_threshold = 1200.0
        self.target_area = 12000.0
        self.close_area = 22000.0
        self.current_twist = Twist()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10,
        )
        self.publish_timer = self.create_timer(0.1, self.publish_velocity)

    def callback(self, data: Image) -> None:
        """Update the robot command based on the latest camera frame."""
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

        green_contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        blue_contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        blue_area = 0.0
        if blue_contours:
            blue_area = cv2.contourArea(max(blue_contours, key=cv2.contourArea))

        command = Twist()
        status = 'Searching for green target'

        if blue_area > self.detect_area_threshold:
            status = 'Blue detected: stopping'
        elif green_contours:
            contour = max(green_contours, key=cv2.contourArea)
            green_area = cv2.contourArea(contour)

            if green_area > self.detect_area_threshold:
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    image_center_x = image.shape[1] // 2
                    error = center_x - image_center_x

                    command.angular.z = float(-0.0025 * error)
                    command.angular.z = max(min(command.angular.z, 0.8), -0.8)

                    if green_area > self.close_area:
                        command.linear.x = -0.05
                        status = 'Green detected: backing away'
                    elif green_area < self.target_area:
                        command.linear.x = 0.08
                        status = 'Green detected: moving forward'
                    else:
                        status = 'Green detected: holding distance'

                    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
                    cv2.circle(
                        image,
                        (int(circle_x), int(circle_y)),
                        int(radius),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(image, (center_x, int(circle_y)), 5, (0, 0, 255), -1)

        self.current_twist = command
        cv2.putText(
            image,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('green_mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('blue_mask', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_feed', image)
        cv2.imshow('green_mask', green_mask)
        cv2.imshow('blue_mask', blue_mask)
        cv2.resizeWindow('camera_feed', 640, 480)
        cv2.resizeWindow('green_mask', 640, 480)
        cv2.resizeWindow('blue_mask', 640, 480)
        cv2.waitKey(1)

    def publish_velocity(self) -> None:
        """Publish the latest velocity command."""
        self.publisher.publish(self.current_twist)

    def stop(self) -> None:
        """Stop the robot immediately."""
        self.current_twist = Twist()
        self.publisher.publish(self.current_twist)


def main(args=None) -> None:
    """Run the colour-following node until interrupted."""
    rclpy.init(args=args)
    robot = Robot()

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
        robot.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
