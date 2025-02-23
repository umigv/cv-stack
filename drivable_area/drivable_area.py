import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array
from std_msgs.msg import Header
import time
from drivable_area.bev import CameraProperties, getBirdView
from skimage.draw import polygon

UNKNOWN = -1
OCCUPIED = 100
FREE = 0
ROBOT = 2

class DrivableArea(Node):
    def __init__(self):
        super().__init__('drivable_area')
        
        # the topic 'url' should be changed to a more specific topic name
        self.time_of_frame = time.time()
        self.subscription = self.create_subscription(
            Image,
            '/zed/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.zed = CameraProperties(63, 68.0, 101.0, 60.0)
        self.curr_pix_size = 0.0055
        self.desired_size = 0.05
        self.image_height = 720
        self.image_width = 1280
        self.table = np.array([((i / 255.0) ** (1.0 / 0.3)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.scale_factor = self.curr_pix_size / self.desired_size
        self.memory_buffer = np.zeros((self.image_height, self.image_width))
        # Create a publisher that publishes OccupancyGrid messages on the 'occupancy_grid' topic
        self.publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)

    def get_occupancy_grid(self, frame):
        image_width, image_height = frame.shape[1], frame.shape[0]
        occupancy_grid = np.zeros((image_height, image_width))
        occupancy_grid_lane = self.update_mask(frame)
        occupancy_grid[occupancy_grid_lane == 255] = 255

        return occupancy_grid
    
    def update_mask(self, image):
        image = cv2.LUT(image, self.table)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 136])
        upper_bound = np.array([179, 36, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1500 # Adjust based on noise size
        contoured = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(contoured, [cnt], -1, 255, thickness=cv2.FILLED)
        
        return contoured
        
        
    def listener_callback(self, msg):
        
        # Convert the ROS message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image to 720x1280
        frame = cv2.resize(frame, (1280, 720))

        # Get the occupancy grid
        occupancy_grid_display = self.get_occupancy_grid(frame)

        # Get the bird's eye view of the occupancy grid
        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occupancy_grid_display, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)


        # Create a mask to remove the area outside the drivable area
        mask = np.full((maxHeight, maxWidth), UNKNOWN, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - 17, bottomRight[1]], [topRight[0] - 75, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        rr, cc = polygon(pts[:, 1], pts[:, 0], mask.shape)
        mask[rr, cc] = FREE

        # Apply the mask to the occupancy grid
        indicies = np.where(mask == UNKNOWN)
        transformed_image[indicies] = UNKNOWN

        # Add a negative border to the occupancy grid
        add_neg = np.full((transformed_image.shape[0], 66), UNKNOWN, dtype=np.int8)

        # Concatenate the negative border to the occupancy grid
        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)
        
        # Convert the occupancy grid to a binary grid with in-place boolean assignments
        binary_grid = np.full(transformed_image.shape, -1, dtype=transformed_image.dtype)
        binary_grid[transformed_image == 255] = OCCUPIED
        binary_grid[transformed_image == 0] = FREE
        transformed_image = binary_grid

        new_size = (int(transformed_image.shape[1] * self.scale_factor), int(transformed_image.shape[0] * self.scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)

        # Create a robot occupancy grid to display the robot's position
        rob_arr = np.full((26, new_size[0]), UNKNOWN, dtype=np.int8)
        rob_arr[13][77] = ROBOT

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))
        combined_arr = np.flipud(combined_arr)
        
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
        # in reality, the pose of the robot should be used here from the zed topics
        grid.info.origin.position.x = 34.0
        grid.info.origin.position.y = 73.0
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