import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array
from std_msgs.msg import Header
import math
import time
from drivable_area.bev import CameraProperties, getBirdView

# Load the YOLO models for lane and pothole detection
lane_model = YOLO('src/drivable_area/drivable_area/utils/LLOnly180ep.pt')
hole_model = YOLO('src/drivable_area/drivable_area/utils/potholesonly100epochs.pt')

UNKNOWN = -1
OCCUPIED = 100
FREE = 0

class DrivableArea(Node):
    def __init__(self):
        super().__init__('drivable_area')
        
        # the topic 'url' should be changed to a more specific topic name
        self.subscription = self.create_subscription(
            Image,
            'zed_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.zed = CameraProperties(54, 68.0, 101.0, 68.0)
        self.curr_pix_size = 0.006
        self.desired_size = 0.05
        self.scale_factor = self.curr_pix_size / self.desired_size

        # Create a publisher that publishes OccupancyGrid messages on the 'occupancy_grid' topic
        self.publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)

    def get_occupancy_grid(self, frame):
        
        # Predict the lane
        r_lane = lane_model.predict(frame, conf=0.5)[0]
        image_width, image_height = frame.shape[1], frame.shape[0]
        
        # Create an empty occupancy grid
        occupancy_grid = np.zeros((image_height, image_width))

        # Predict the potholes
        r_hole = hole_model.predict(frame, conf=0.25)[0]

        # If the lane is detected, fill the occupancy grid with the lane and mark the undrivable area as occupied
        time_of_frame = 0
        if r_lane.masks is not None:
            if(len(r_lane.masks.xy) != 0):
                segment = r_lane.masks.xy[0]
                segment_array = np.array([segment], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [segment_array], 255)
                time_of_frame = time.time()

        # If the potholes are detected, put a mask of the potholes on the occupancy grid and mark the area as occupied
        if r_hole.boxes is not None:
            for segment in r_hole.boxes.xyxyn:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([[x_min*image_width, y_min*image_height], 
                                    [x_max*image_width, y_min*image_height], 
                                    [x_max*image_width, y_max*image_height], 
                                    [x_min*image_width, y_max*image_height]], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))

        # Calculate the buffer time
        buffer_area = np.sum(occupancy_grid)//255
        buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)
        return occupancy_grid, buffer_time, time_of_frame
        
    def listener_callback(self, msg):
        
        # Convert the ROS message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image to 720x1280
        frame = cv2.resize(frame, (1280, 720))

        # Get the occupancy grid
        occupancy_grid_display, buffer_time, time_of_frame = self.get_occupancy_grid(frame)
        total = np.sum(occupancy_grid_display)
        curr_time = time.time()

        # If the occupancy grid is undetectable, display the previous frame
        if total == 0:
            if curr_time - time_of_frame < buffer_time:
                occupancy_grid_display = memory_buffer
            else:
                occupancy_grid_display.fill(255)
        else:
            memory_buffer = occupancy_grid_display

        # Get the bird's eye view of the occupancy grid
        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occupancy_grid_display, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)


        # Create a mask to remove the area outside the drivable area
        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - 27, bottomRight[1]], [topRight[0] - 65, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        pts = pts.reshape((-1, 1, 2))  # reshape points
        cv2.fillPoly(mask, [pts], True, 0)

        # Apply the mask to the occupancy grid
        indicies = np.where(mask == -1)
        transformed_image[indicies] = -1

        # Add a negative border to the occupancy grid
        add_neg = np.full((transformed_image.shape[0], 66), -1, dtype=np.int8)

        # Concatenate the negative border to the occupancy grid
        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)
        
        # Convert the occupancy grid to a binary grid
        transformed_image = np.where(transformed_image==255, 1, transformed_image)
        transformed_image = np.where((transformed_image != 0) & (transformed_image != 1) & (transformed_image != -1), -1, transformed_image)

        new_size = (int(transformed_image.shape[1] * self.scale_factor), int(transformed_image.shape[0] * self.scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)

        # Create a robot occupancy grid to display the robot's position
        rob_arr = np.full((22, 169), -1, dtype=np.int8)
        rob_arr[10][85] = 2

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))

        combined_arr = np.where(combined_arr==0, 3, combined_arr)
        combined_arr = np.where(combined_arr==1, 0, combined_arr)
        combined_arr = np.where(combined_arr==3, 1, combined_arr)

        # np.savetxt('occupancy_grid.txt', combined_arr, fmt='%d')
        
        self.send_occupancy_grid(combined_arr)

    def send_occupancy_grid(self, array):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'
        grid.info = MapMetaData()
        grid.info.resolution = self.desired_size
        grid.info.width = array.shape[1]
        grid.info.height = array.shape[0]
        grid.info.origin.position.x = 34.0
        grid.info.origin.position.y = 85.0
        grid.info.origin.position.z = 0.0

        grid.data = Array('b', array.ravel().astype(np.int8))

        self.publisher.publish(grid)
        self.get_logger().info('Publishing occupancy grid')


def main(args=None):
    rclpy.init(args=args)

    node = DrivableArea()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()