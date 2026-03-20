import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class FSMRobot(Node):
    def __init__(self):
        super().__init__("fsm_robot")
        self.bridge = CvBridge()
        self.state = "EXPLORE"
        self.get_logger().info("Starting in state: EXPLORE")

        self.create_subscription(LaserScan, "/laser/scan", self.lidar_cb, 10)
        self.create_subscription(Image, "/camera/image_raw", self.camera_cb, 10)
        self.create_subscription(Imu, "/imu", self.imu_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.min_lidar = float("inf")
        self.red_detected = False
        self.red_cx = 320
        self.imu_stop = False

        self.create_timer(0.1, self.control_loop)

    def lidar_cb(self, msg):
        ranges = [r for r in msg.ranges if not np.isnan(r) and not np.isinf(r)]
        self.min_lidar = min(ranges) if ranges else float("inf")

    def camera_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 120, 70]),   np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 200:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        self.red_cx = int(M["m10"] / M["m00"])
                    self.red_detected = True
                    return
            self.red_detected = False
        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")
            self.red_detected = False

    def imu_cb(self, msg):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        self.imu_stop = abs(ax) > 8.0 or abs(ay) > 8.0 or abs(wx) > 2.5 or abs(wy) > 2.5 or abs(wz) > 2.5

    def set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"State: {self.state} --> {new_state}")
            self.state = new_state

    def control_loop(self):
        twist = Twist()

        if self.imu_stop:
            self.set_state("STOP")
        elif self.min_lidar < 0.6:
            self.set_state("AVOID")
        elif self.state == "AVOID" and self.min_lidar >= 0.6:
            self.set_state("EXPLORE")
        elif self.state == "EXPLORE" and self.red_detected:
            self.set_state("TRACK")
        elif self.state == "TRACK" and not self.red_detected:
            self.set_state("EXPLORE")
        elif self.state == "STOP" and not self.imu_stop:
            self.set_state("EXPLORE")

        if self.state == "EXPLORE":
            twist.linear.x = 0.0
            twist.angular.z = 0.3
        elif self.state == "TRACK":
            error = self.red_cx - 320
            twist.linear.x = 0.3
            twist.angular.z = -float(error) / 640.0
        elif self.state == "AVOID":
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        elif self.state == "STOP":
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FSMRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
