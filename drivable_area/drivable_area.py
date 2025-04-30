import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped
from array import array as Array
from std_msgs.msg import Header
from drivable_area.bev import CameraProperties, getBirdView
from skimage.draw import polygon
from ultralytics import YOLO
from drivable_area.occ_grid import occ_grid
from drivable_area.hsv import hsv

UNKNOWN = -1
OCCUPIED = 100
FREE = 0
ROBOT = 2

class DrivableArea(Node):
    def __init__(self):
        super().__init__('drivable_area')
        # Declare parameters for configuration
        self.declare_parameter('image.width', 1280)
        self.declare_parameter('image.height', 720)
        self.declare_parameter('image.left_border', 66)
        self.declare_parameter('robot.grid_height', 26)
        self.declare_parameter('robot.row', 13)
        self.declare_parameter('robot.col', 77)
        self.declare_parameter('grid.position.x', 34.0)
        self.declare_parameter('grid.position.y', 73.0)
        self.declare_parameter('grid.position.z', 0.0)
        self.declare_parameter('camera.zed_height', 63)
        self.declare_parameter('camera.zed_fov_vert', 68.0)
        self.declare_parameter('camera.zed_fov_horz', 101.0)
        self.declare_parameter('camera.zed_camera_tilt', 60.0)
        self.declare_parameter('topics.image_subscription', '/zed/image_raw')
        self.declare_parameter('topics.occupancy_grid', 'test_occ')
        self.declare_parameter('sizes.desired_size', 0.05)
        self.declare_parameter('sizes.curr_pix_size', 0.0055)
        self.declare_parameter('offsets.polygon_offset_right', 17)
        self.declare_parameter('offsets.polygon_offset_top', 75)
        # New: Declare HSV filter parameters as arrays
        self.declare_parameter('hsv.lower', [0, 0, 136])
        self.declare_parameter('hsv.upper', [179, 36, 255])
        self.declare_parameter('morph.iterations', 2)
        # YOLO parameters
        self.declare_parameter('yolo.lane_model_path', 'src/cv-stack/drivable_area/utils/laneswithcontrast.pt')
        self.declare_parameter('yolo.hole_model_path', 'src/cv-stack/drivable_area/utils/potholesonly100epochs.pt')
        self.declare_parameter('yolo.lane_confidence', 0.5)
        self.declare_parameter('yolo.hole_confidence', 0.25)
        self.declare_parameter('yolo.lane_extension', 35)
        self.declare_parameter('yolo.lane_search_height', 100)
        
        # Build config from parameters
        self.config = {
            "image": {
                "width": self.get_parameter('image.width').value,
                "height": self.get_parameter('image.height').value,
                "left_border": self.get_parameter('image.left_border').value
            },
            "robot": {
                "grid_height": self.get_parameter('robot.grid_height').value,
                "row": self.get_parameter('robot.row').value,
                "col": self.get_parameter('robot.col').value
            },
            "grid": {
                "position": {
                    "x": self.get_parameter('grid.position.x').value,
                    "y": self.get_parameter('grid.position.y').value,
                    "z": self.get_parameter('grid.position.z').value
                }
            },
            "camera": {
                "zed_height": self.get_parameter('camera.zed_height').value,
                "zed_fov_vert": self.get_parameter('camera.zed_fov_vert').value,
                "zed_fov_horz": self.get_parameter('camera.zed_fov_horz').value,
                "zed_camera_tilt": self.get_parameter('camera.zed_camera_tilt').value
            },
            "topics": {
                "image_subscription": self.get_parameter('topics.image_subscription').value,
                "occupancy_grid": self.get_parameter('topics.occupancy_grid').value
            },
            "sizes": {
                "desired_size": self.get_parameter('sizes.desired_size').value,
                "curr_pix_size": self.get_parameter('sizes.curr_pix_size').value
            },
            "offsets": {
                "polygon_offset_right": self.get_parameter('offsets.polygon_offset_right').value,
                "polygon_offset_top": self.get_parameter('offsets.polygon_offset_top').value
            },
            "hsv": {
                "lower": self.get_parameter('hsv.lower').value,
                "upper": self.get_parameter('hsv.upper').value,
            },
            "morph": {
                "iterations": self.get_parameter('morph.iterations').value
            },
            "yolo": {
                "lane_model_path": self.get_parameter('yolo.lane_model_path').value,
                "hole_model_path": self.get_parameter('yolo.hole_model_path').value,
                "lane_confidence": self.get_parameter('yolo.lane_confidence').value,
                "hole_confidence": self.get_parameter('yolo.hole_confidence').value,
                "lane_extension": self.get_parameter('yolo.lane_extension').value,
                "lane_search_height": self.get_parameter('yolo.lane_search_height').value
            }
        }
        
        # Load YOLO models
        self.get_logger().info(f"Loading YOLO models from {self.config['yolo']['lane_model_path']} and {self.config['yolo']['hole_model_path']}")
        self.lane_model = YOLO(self.config["yolo"]["lane_model_path"])
        self.hole_model = YOLO(self.config["yolo"]["hole_model_path"])

        self.image_width = self.config["image"]["width"]
        self.image_height = self.config["image"]["height"]
        self.subscription = self.create_subscription(
            Image,
            self.config["topics"]["image_subscription"],
            self.listener_callback,
            10)
        
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10
        )
        self.pose_subscription  # Prevent unused variable warning

        self.robot_position_x = float(0)
        self.robot_position_y = float(0)
        self.robot_position_z = float(0)
        self.robot_orientation = float(0)

        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.zed = CameraProperties(
            self.config["camera"]["zed_height"],
            self.config["camera"]["zed_fov_vert"],
            self.config["camera"]["zed_fov_horz"],
            self.config["camera"]["zed_camera_tilt"])
        self.curr_pix_size = self.config["sizes"]["curr_pix_size"]
        self.desired_size = self.config["sizes"]["desired_size"]
        self.table = np.array([((i / 255.0) ** (1.0 / 0.3)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.scale_factor = self.curr_pix_size / self.desired_size
        self.memory_buffer = np.zeros((self.image_height, self.image_width))
        # Create a publisher that publishes OccupancyGrid messages on the 'occupancy_grid' topic
        self.publisher = self.create_publisher(OccupancyGrid, self.config["topics"]["occupancy_grid"], 10)
        self.final_mask = None
        self.occ_obj = occ_grid(0, self.config["yolo"]["lane_model_path"]) #occ_grid(video_path, yolo_path)
        self.hsv_obj = hsv(0)


    def get_occupancy_grid(self, frame):
        image_width, image_height = frame.shape[1], frame.shape[0]
        # occupancy_grid = np.zeros((image_height, image_width))
        
        # HSV-based detection
        combined, dict = self.hsv_obj.get_mask(frame)
        # print(dict)
        # print("HSV mask shape: ", combined.shape)
        # Extend lanes at the bottom of the image
        lane_search_height = self.config["yolo"]["lane_search_height"]
        lane_extension = self.config["yolo"]["lane_extension"]
        for i in range(combined.shape[1]):
            if np.any(combined[-lane_search_height:, i]):
                combined[-lane_extension:, i] = 255
        
        # YOLO pothole detection
        r_hole = self.hole_model.predict(frame, conf=self.config["yolo"]["hole_confidence"], device='cpu')[0]
        
        if r_hole.boxes is not None:
            for segment in r_hole.boxes.xyxyn:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([
                    [x_min * image_width, y_min * image_height],
                    [x_max * image_width, y_min * image_height],
                    [x_max * image_width, y_max * image_height],
                    [x_min * image_width, y_max * image_height]
                ], dtype=np.int32)
                cv2.fillPoly(combined, [vertices], color=255)
        
        return combined
    
    def pose_callback(self, msg):
        # Extract the position and orientation from the pose message
        position = msg.pose.position
        orientation = msg.pose.orientation

        # Convert the position to grid coordinates (optional: you may need to apply a transformation if your map frame is different)
        self.robot_position_x = position.x
        self.robot_position_y = position.y
        self.robot_position_z = position.z  # If needed for 3D grids

        self.robot_orientation = orientation
        
        
    def listener_callback(self, msg):
        
        # Convert the ROS message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image to 720x1280
        frame = cv2.resize(frame, (self.config["image"]["width"], self.config["image"]["height"]))

        # Get the occupancy grid
        occupancy_grid_display = self.get_occupancy_grid(frame)

        # Get the bird's eye view of the occupancy grid
        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occupancy_grid_display, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)


        # Create a mask to remove the area outside the drivable area
        mask = np.full((maxHeight, maxWidth), UNKNOWN, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - self.config["offsets"]["polygon_offset_right"], bottomRight[1]], [topRight[0] - self.config["offsets"]["polygon_offset_top"], topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        rr, cc = polygon(pts[:, 1], pts[:, 0], mask.shape)
        mask[rr, cc] = FREE

        # Apply the mask to the occupancy grid
        indicies = np.where(mask == UNKNOWN)
        transformed_image[indicies] = UNKNOWN

        # Add a negative border to the occupancy grid
        add_neg = np.full((transformed_image.shape[0], self.config["image"]["left_border"]), UNKNOWN, dtype=np.int8)

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
        rob_arr = np.full((self.config["robot"]["grid_height"], new_size[0]), UNKNOWN, dtype=np.int8)
        rob_arr[self.config["robot"]["row"]][self.config["robot"]["col"]] = ROBOT

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))
        # combined_arr = np.flipud(combined_arr)
        combined_arr = np.rot90(combined_arr, -1)
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
        # grid.info.origin.position.x = self.config["grid"]["position"]["x"]
        # grid.info.origin.position.y = self.config["grid"]["position"]["y"]
        # grid.info.origin.position.z = self.config["grid"]["position"]["z"]

        grid.info.origin.position.x = self.robot_position_x
        grid.info.origin.position.y = self.robot_position_y
        # grid.info.origin.position.z = self.robot_position_z
        # grid.info.origin.orientation = self.robot_orientation

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