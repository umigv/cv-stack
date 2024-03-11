import cv2
import numpy as np
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge


video = cv2.VideoCapture("drivable_area/drivable_area/utils/comp23_2.mp4")


fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height), isColor=True)
# out2 = cv2.VideoWriter('occupancy_grid.avi', fourcc, fps, (width, height), isColor=True)

class MinimalPublisher(Node):

    def __init__(self):
        # Call the constructor of the parent class Node
        super().__init__('minimal_publisher')
        
        self.bridge = CvBridge()
        # Create a publisher that publishes OccupancyGrid messages on the 'topic' topic
        self.publisher_ = self.create_publisher(Image, 'zed_image', 10)
        # made the timer period the same as the FPS
        timer_period = 1 / fps  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        
        ret, frame = video.read()
        if not ret:
            return
        # Publish the occupancy grid
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(ros_image)
        # Log a message
        self.get_logger().info('Publishing ZED frame')
        
        
        
# The main function
def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    # Create a MinimalPublisher object
    minimal_publisher = MinimalPublisher()

    # Spin the node so it can process callbacks
    rclpy.spin(minimal_publisher)

    # After spinning, destroy the node and shutdown rclpy
    minimal_publisher.destroy_node()
    rclpy.shutdown()

# If this script is run directly, call the main function
if __name__ == '__main__':
    main()