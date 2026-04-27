"""Integrated project node for nav2 waypoint driving and blue-box seeking."""

from math import atan2, cos, isfinite, sin

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float between lower and upper bounds."""
    return max(lower, min(value, upper))


class BoxSeeker(Node):
    """Navigate with nav2 from an RViz goal, then search for and approach the blue box."""

    def __init__(self) -> None:
        super().__init__('box_seeker')

        self.declare_parameter('target_x', 0.33)
        self.declare_parameter('target_y', -9.25)
        self.declare_parameter('target_yaw', 0.0)
        self.declare_parameter('use_rviz_goal', False)

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10,
        )
        self.rviz_goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10,
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10,
        )
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
        )
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.target_yaw = float(self.get_parameter('target_yaw').value)
        self.use_rviz_goal = bool(self.get_parameter('use_rviz_goal').value)
        self.min_contour_area = 400.0
        self.search_turn_speed = 0.7
        self.max_turn_speed = 1.6
        self.approach_speed = 0.24
        self.slow_approach_speed = 0.10
        self.stop_distance = 0.9
        self.close_distance = 1.4
        self.front_sector_half_angle = np.deg2rad(10.0)
        self.last_turn_direction = 1.0

        self.state = 'WAIT_FOR_GOAL' if self.use_rviz_goal else 'NAVIGATE_TO_WAYPOINT'
        self.latest_detections = {}
        self.latest_frame_width = None
        self.front_distance = None
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.goal_received = not self.use_rviz_goal
        self.seen_colours = set()
        self.required_colours = {'red', 'green', 'blue'}
        self.all_colours_logged = False
        self.navigation_goal_sent = False
        self.navigation_goal_complete = False
        self.navigation_goal_handle = None
        self.navigation_cancel_requested = False
        self.current_command = Twist()
        self.last_status = ''

        self.get_logger().info(
            'Box seeker ready: '
            + (
                'waiting for an RViz nav goal before starting'
                if self.use_rviz_goal
                else f'navigating to target ({self.target_x:.2f}, {self.target_y:.2f}) with nav2 before scanning for blue'
            )
        )

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
        self.update_seen_colours(detections)

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

    def odom_callback(self, msg: Odometry) -> None:
        """Track the robot pose from odometry for overlays and local approach."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.current_x = position.x
        self.current_y = position.y

        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.current_yaw = atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg: PoseStamped) -> None:
        """Accept a new RViz goal and start the nav2 phase from that target."""
        if msg.header.frame_id and msg.header.frame_id != 'map':
            self.get_logger().warn(
                f'Ignoring goal in frame {msg.header.frame_id}; expected map frame'
            )
            return

        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y

        orientation = msg.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.target_yaw = atan2(siny_cosp, cosy_cosp)

        self.goal_received = True
        self.navigation_goal_complete = False
        self.navigation_goal_sent = False
        self.navigation_cancel_requested = False
        if self.navigation_goal_handle is not None:
            self.cancel_navigation_goal()

        self.state = 'NAVIGATE_TO_WAYPOINT'
        self.log_state(
            f'Received RViz goal at ({self.target_x:.2f}, {self.target_y:.2f}), starting nav2'
        )

    def clicked_point_callback(self, msg: PointStamped) -> None:
        """Accept an RViz clicked point and convert it into a nav2 target."""
        if msg.header.frame_id and msg.header.frame_id != 'map':
            self.get_logger().warn(
                f'Ignoring clicked point in frame {msg.header.frame_id}; expected map frame'
            )
            return

        self.target_x = msg.point.x
        self.target_y = msg.point.y

        if self.current_x is not None and self.current_y is not None:
            self.target_yaw = atan2(self.target_y - self.current_y, self.target_x - self.current_x)
        else:
            self.target_yaw = 0.0

        self.goal_received = True
        self.navigation_goal_complete = False
        self.navigation_goal_sent = False
        self.navigation_cancel_requested = False
        if self.navigation_goal_handle is not None:
            self.cancel_navigation_goal()

        self.state = 'NAVIGATE_TO_WAYPOINT'
        self.log_state(
            f'Received RViz clicked point at ({self.target_x:.2f}, {self.target_y:.2f}), starting nav2'
        )

    def update_seen_colours(self, detections) -> None:
        """Remember which target colours have been seen at least once."""
        new_colours = sorted(set(detections.keys()) - self.seen_colours)
        for colour_name in new_colours:
            self.seen_colours.add(colour_name)
            self.get_logger().info(f'Seen {colour_name} box')

        if self.all_colours_seen() and not self.all_colours_logged:
            self.get_logger().info('All RGB boxes observed')
            self.all_colours_logged = True

    def all_colours_seen(self) -> bool:
        """Return True once all required colours have been observed."""
        return self.required_colours.issubset(self.seen_colours)

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
        seen_text = 'seen=' + ', '.join(sorted(self.seen_colours)) if self.seen_colours else 'seen=none'
        pose_text = (
            f'pose=({self.current_x:.2f}, {self.current_y:.2f}) '
            f'target=({self.target_x:.2f}, {self.target_y:.2f})'
            if self.current_x is not None and self.current_y is not None
            else f'target=({self.target_x:.2f}, {self.target_y:.2f})'
        )
        goal_text = 'goal=ready' if self.goal_received else 'goal=waiting'
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
        cv2.putText(
            image,
            pose_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            seen_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            goal_text,
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def send_navigation_goal(self) -> None:
        """Send the nav2 goal for the exploration waypoint."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.target_x
        goal_msg.pose.pose.position.y = self.target_y
        goal_msg.pose.pose.orientation.z = sin(self.target_yaw / 2.0)
        goal_msg.pose.pose.orientation.w = cos(self.target_yaw / 2.0)

        self.navigation_goal_sent = True
        send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback,
        )
        send_goal_future.add_done_callback(self.navigation_goal_response_callback)
        self.log_state('Sent nav2 goal to exploration waypoint')

    def navigation_feedback_callback(self, feedback_msg) -> None:
        """Keep the action interface connected while nav2 is exploring."""
        del feedback_msg

    def navigation_goal_response_callback(self, future) -> None:
        """Handle acceptance or rejection of the nav2 goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.navigation_goal_sent = False
            self.navigation_goal_handle = None
            self.log_state('Nav2 goal rejected, retrying')
            return

        self.navigation_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future) -> None:
        """Record completion of the nav2 goal."""
        if self.state == 'NAVIGATE_TO_WAYPOINT':
            self.navigation_goal_complete = True
            self.log_state('Nav2 waypoint reached, switching to search')
        self.navigation_goal_sent = False
        self.navigation_goal_handle = None
        self.navigation_cancel_requested = False
        del future

    def cancel_navigation_goal(self) -> None:
        """Cancel the active nav2 goal so visual servoing can take over."""
        if self.navigation_goal_handle is None or self.navigation_cancel_requested:
            return

        self.navigation_cancel_requested = True
        cancel_future = self.navigation_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.navigation_cancel_callback)
        self.log_state('Cancelling nav2 goal to start blue approach')

    def navigation_cancel_callback(self, future) -> None:
        """Clear nav2 goal state after a cancellation request."""
        self.navigation_goal_sent = False
        self.navigation_goal_handle = None
        self.navigation_cancel_requested = False
        del future

    def control_loop(self) -> None:
        """Update robot motion based on the latest detections and lidar data."""
        detections = self.latest_detections
        blue_detection = detections.get('blue')
        command = Twist()
        publish_command = True

        if self.state == 'WAIT_FOR_GOAL':
            if self.goal_received:
                self.state = 'NAVIGATE_TO_WAYPOINT'
            else:
                self.log_state('Waiting for RViz goal on /move_base_simple/goal or /goal_pose')

        if self.state == 'NAVIGATE_TO_WAYPOINT':
            publish_command = False
            if blue_detection is not None and self.all_colours_seen():
                self.cancel_navigation_goal()
                self.state = 'APPROACH_BLUE'
                publish_command = True
                self.log_state('Blue detected after all colours seen, approaching target')
            elif self.navigation_goal_complete:
                self.state = 'SEARCH_BLUE'
                publish_command = True
            elif not self.goal_received:
                self.state = 'WAIT_FOR_GOAL'
                publish_command = True
            elif not self.navigation_goal_sent and not self.navigation_cancel_requested:
                if self.nav_action_client.wait_for_server(timeout_sec=0.1):
                    self.send_navigation_goal()
                else:
                    self.log_state('Waiting for nav2 action server')

        if self.state == 'SEARCH_BLUE':
            if blue_detection is not None and self.all_colours_seen():
                self.state = 'APPROACH_BLUE'
                self.log_state('Blue detected after all colours seen, approaching target')
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

        if publish_command:
            self.current_command = command
            self.cmd_pub.publish(self.current_command)

    def log_state(self, message: str) -> None:
        """Avoid spamming repeated log messages."""
        if message != self.last_status:
            self.get_logger().info(message)
            self.last_status = message

    def stop(self) -> None:
        """Stop the robot and publish a zero twist."""
        self.cancel_navigation_goal()
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
