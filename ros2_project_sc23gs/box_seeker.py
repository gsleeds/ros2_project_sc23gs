"""Integrated project node for RGB box detection and blue-box seeking."""

from math import isfinite

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float between lower and upper bounds."""
    return max(lower, min(value, upper))


class BoxSeeker(Node):
    """Detect RGB boxes, search for blue, approach it, and stop nearby."""

    def __init__(self) -> None:
        super().__init__('box_seeker')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10,
        )
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.min_contour_area = 400.0
        self.search_turn_speed = 0.35
        self.max_turn_speed = 0.8
        self.approach_speed = 0.12
        self.slow_approach_speed = 0.05
        self.stop_distance = 0.7
        self.close_distance = 1.1
        self.front_sector_half_angle = np.deg2rad(10.0)
        self.last_turn_direction = 1.0

        self.state = 'SEARCH_BLUE'
        self.latest_detections = {}
        self.latest_frame_width = None
        self.front_distance = None
        self.current_command = Twist()
        self.last_status = ''

        self.get_logger().info('Box seeker ready: searching for blue box')

    def image_callback(self, msg: Image) -> None:
        """Process the latest camera frame and update colour detections."""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        self.latest_frame_width = image.shape[1]
        detections, combined_mask = self.detect_colours(image)
        self.latest_detections = detections

        annotated_image = image.copy()
        self.draw_detections(annotated_image, detections)
        self.draw_status(annotated_image, detections)

        cv2.namedWindow('box_seeker_view', cv2.WINDOW_NORMAL)
        cv2.namedWindow('box_seeker_mask', cv2.WINDOW_NORMAL)
        cv2.imshow('box_seeker_view', annotated_image)
        cv2.imshow('box_seeker_mask', combined_mask)
        cv2.resizeWindow('box_seeker_view', 960, 720)
        cv2.resizeWindow('box_seeker_mask', 640, 480)
        cv2.waitKey(1)

    def scan_callback(self, msg: LaserScan) -> None:
        """Track the minimum valid lidar distance in front of the robot."""
        front_ranges = []

        for index, distance in enumerate(msg.ranges):
            if not isfinite(distance) or distance <= 0.0:
                continue

            angle = msg.angle_min + index * msg.angle_increment
            normalized_angle = np.arctan2(np.sin(angle), np.cos(angle))
            if abs(normalized_angle) <= self.front_sector_half_angle:
                front_ranges.append(distance)

        self.front_distance = min(front_ranges) if front_ranges else None

    def detect_colours(self, image):
        """Return the largest contour for each target colour and a combined mask."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_mask_low = cv2.inRange(
            hsv_image,
            np.array([0, 120, 80]),
            np.array([10, 255, 255]),
        )
        red_mask_high = cv2.inRange(
            hsv_image,
            np.array([170, 120, 80]),
            np.array([179, 255, 255]),
        )
        masks = {
            'red': cv2.bitwise_or(red_mask_low, red_mask_high),
            'green': cv2.inRange(
                hsv_image,
                np.array([45, 80, 80]),
                np.array([85, 255, 255]),
            ),
            'blue': cv2.inRange(
                hsv_image,
                np.array([100, 120, 80]),
                np.array([135, 255, 255]),
            ),
        }

        kernel = np.ones((5, 5), np.uint8)
        detections = {}
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for colour_name, mask in masks.items():
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.bitwise_or(combined_mask, cleaned_mask)

            contours, _ = cv2.findContours(
                cleaned_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            detections[colour_name] = {
                'area': area,
                'center_x': x + width / 2.0,
                'center_y': y + height / 2.0,
                'bbox': (x, y, width, height),
                'mask': cleaned_mask,
            }

        return detections, combined_mask

    def draw_detections(self, image, detections) -> None:
        """Draw bounding boxes and labels for all visible colours."""
        colours_bgr = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
        }

        for colour_name, detection in detections.items():
            x, y, width, height = detection['bbox']
            draw_colour = colours_bgr[colour_name]
            label = f'{colour_name} area={int(detection["area"])}'
            cv2.rectangle(image, (x, y), (x + width, y + height), draw_colour, 2)
            cv2.putText(
                image,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_colour,
                2,
            )

    def draw_status(self, image, detections) -> None:
        """Overlay node state and range information on the live image."""
        visible_colours = ', '.join(sorted(detections.keys())) or 'none'
        range_text = (
            f'front_range={self.front_distance:.2f}m'
            if self.front_distance is not None
            else 'front_range=unknown'
        )
        status_text = f'state={self.state} visible={visible_colours}'
        cv2.putText(
            image,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            range_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def control_loop(self) -> None:
        """Update robot motion based on the latest detections and lidar data."""
        detections = self.latest_detections
        blue_detection = detections.get('blue')
        command = Twist()

        if self.state == 'SEARCH_BLUE':
            if blue_detection is not None:
                self.state = 'APPROACH_BLUE'
                self.log_state('Blue detected, approaching target')
            else:
                command.angular.z = self.search_turn_speed * self.last_turn_direction

        if self.state == 'APPROACH_BLUE':
            if blue_detection is None or self.latest_frame_width is None:
                self.state = 'SEARCH_BLUE'
                self.log_state('Blue lost, resuming search')
                command.angular.z = self.search_turn_speed * self.last_turn_direction
            else:
                center_error = blue_detection['center_x'] - (self.latest_frame_width / 2.0)
                self.last_turn_direction = -1.0 if center_error < 0.0 else 1.0
                command.angular.z = clamp(-0.003 * center_error, -self.max_turn_speed, self.max_turn_speed)

                if self.front_distance is not None and self.front_distance <= self.stop_distance:
                    self.state = 'STOPPED'
                    self.log_state('Blue target reached, stopping')
                else:
                    if abs(center_error) < 40.0:
                        if self.front_distance is not None and self.front_distance <= self.close_distance:
                            command.linear.x = self.slow_approach_speed
                        else:
                            command.linear.x = self.approach_speed

        if self.state == 'STOPPED':
            command = Twist()

        self.current_command = command
        self.cmd_pub.publish(self.current_command)

    def log_state(self, message: str) -> None:
        """Avoid spamming repeated log messages."""
        if message != self.last_status:
            self.get_logger().info(message)
            self.last_status = message

    def stop(self) -> None:
        """Stop the robot and publish a zero twist."""
        self.current_command = Twist()
        self.cmd_pub.publish(self.current_command)


def main(args=None) -> None:
    """Run the integrated box seeker node until interrupted."""
    rclpy.init(args=args)
    node = BoxSeeker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
