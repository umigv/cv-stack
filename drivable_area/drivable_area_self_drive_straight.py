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
from drivable_area.left_turn_new import leftTurn
# from drivable_area.right_turn import right_turn
from map_interfaces.srv import InflationGrid
from infra_interfaces.action import NavigateToGoal
from infra_interfaces.msg import CellCoordinateMsg
from rclpy.action import ActionClient


import os

UNKNOWN = -1
OCCUPIED = 100
FREE = 0
ROBOT = 2

is_left_turn = True
is_right_turn  = False

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
        
        #/zed/zed_node/pose
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'odom',
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

        self.waypoints = None
        self.response = None
        self.inflation_client = self.create_client(InflationGrid, 'inflation_grid_service')
        self.navigate_client = ActionClient(self, NavigateToGoal, 'navigate_to_goal')
        self.response_waypoint = None
        self.processing_navigation = False
        self.barrel_midpoint = None
        self.barrel_boxes


    def get_occupancy_grid(self, frame):
        combined, dict = self.hsv_obj.get_mask(frame)
        lane_search_height = self.config["yolo"]["lane_search_height"]
        lane_extension = self.config["yolo"]["lane_extension"]
        for i in range(combined.shape[1]):
            if np.any(combined[-lane_search_height:, i]):
                combined[-lane_extension:, i] = 255

        # TODO: init left turn object, masks, etc. 
        # remember to update bools
        self.barrel_boxes = self.hsv_obj.barrel_boxes
        for segment in self.barrel_boxes:
            x_min, y_min, x_max, y_max = segment
            vertices = np.array([
                [x_min * self.width, y_min * self.height], #top-left
                [x_max * self.width, y_min * self.height], #top right
                [x_max * self.width, y_max * self.height], #bottom-right
                [x_min * self.width, y_max * self.height] #bottom left
            ], dtype=np.int32)
            
            if(y_min * self.height > self.height // 2):
                # this might be a cone that is close to us so see if its in the midele
                midpoint = (x_max * self.width) - (x_min * self.width)
                if(midpoint > self.width // 4 and midpoint < (self.width - (self.width//4))):
                    self.in_state_4 = True
                    self.barrel_midpoint = midpoint
            else:
                self.barrel_midpoint = None
        
        new = np.array([self.barrel_midpoint[0], self.barrel_midpoint[1], 1])
        multiplied_waypoint = self.zed.matrix @ new
        multiplied_waypoint /= multiplied_waypoint[2]
        tx, ty = multiplied_waypoint[0], multiplied_waypoint[1]
        tx *= self.scale_factor
        ty *= self.scale_factor
        
        self.barrel_midpoint = (tx, ty)
        # return combined occupancy grid and waypoints obtained from turn?
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
    
    def stop(self, occ_grid):
        # r_height, r_width = occ_grid.shape
        # center_width = r_width // 2

        # dist_px = round(3 * .3048 / 0.05) # ~18 pixels AKA 3 feet from bottom

        # start_y = r_height - 1
        # end_y = max(0, r_height - dist_px - 1)

        # # for each pixel within 18 pixels, search for obstacle
        # for y in range(start_y, end_y - 1, -1):
        #     px_val = occ_grid[y, center_width]
        #     if px_val == 100:
        #         return True
        
        print(f"Barrel Midpoint {self.barrel_midpoint}")
        if self.barrel_midpoint > 20:
            return False
        else:
            return True
        
    def write_to_temp(self):
        # write 0 to stop, else 1
        #write a 0 to the temp file reference this: https://github.com/umigv/embedded_ros_marvin/blob/main/sdr_estop/estop_epy_block_3.py
        #use the self.file_path in the embedded file
        #use lines 37-40
        #TODO
        with open(self.file_path, 'w') as f:
            f.write('0')
            f.flush()
            os.fsync(f.fileno())
        return None
        
    def listener_callback(self, msg):
        
        # Convert the ROS message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image to 720x1280
        frame = cv2.resize(frame, (self.config["image"]["width"], self.config["image"]["height"]))

        print(f"awaiting_navigate_response: {self.processing_navigation}")
        if self.processing_navigation:
            return
        
        self.processing_navigation = True

        # TODO: changed to return waypoints too
        # Get the occupancy grid and waypoints from turn 
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
        
        #TODO: Multiply best waypoint by ZED matrix
        

        self.response_waypoint = (77, 52)

        # Convert the occupancy grid to a binary grid with in-place boolean assignments
        binary_grid = np.full(transformed_image.shape, -1, dtype=transformed_image.dtype)
        binary_grid[transformed_image == 255] = OCCUPIED
        binary_grid[transformed_image == 0] = FREE
        transformed_image = binary_grid

        new_size = (int(transformed_image.shape[1] * self.scale_factor), int(transformed_image.shape[0] * self.scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)
        print(f"Resized image shape: {resized_image.shape}")


        should_stop = self.stop(resized_image)

        if should_stop:
            self.write_to_temp()
            
            
            
        # Create a robot occupancy grid to display the robot's position
        rob_arr = np.full((self.config["robot"]["grid_height"], new_size[0]), UNKNOWN, dtype=np.int8)
        rob_arr[self.config["robot"]["row"]][self.config["robot"]["col"]] = ROBOT

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))
        # combined_arr = np.flipud(combined_arr)
        combined_arr = np.rot90(combined_arr, -1)
        combined_arr = np.flipud(combined_arr)
        
        self.send_occupancy_grid(combined_arr)
        self.send_inflation_request()

    def send_occupancy_grid(self, array):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'odom'
        grid.info = MapMetaData()
        grid.info.resolution = self.desired_size
        grid.info.width = array.shape[1]
        grid.info.height = array.shape[0]
        # in reality, the pose of the robot should be used here from the zed topics
        # grid.info.origin.position.x = self.config["grid"]["position"]["x"]
        # grid.info.origin.position.y = self.config["grid"]["position"]["y"]
        # grid.info.origin.position.z = self.config["grid"]["position"]["z"]

        grid.info.origin.position.x = -14.0 * grid.info.resolution
        grid.info.origin.position.y = -76 * grid.info.resolution
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.x = 0.0
        grid.info.origin.orientation.y = 0.0
        grid.info.origin.orientation.z = 0.0
        grid.info.origin.orientation.w = 1.0

        grid.data = Array('b', array.ravel().astype(np.int8))

        self.publisher.publish(grid)
        self.get_logger().info('Publishing occupancy grid')

    def send_inflation_request(self):
        while not self.inflation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for inflation_grid_service...')

        self.get_logger().info("Sending inflation_grid_service request")
        self.req = InflationGrid.Request()
        future = self.inflation_client.call_async(self.req)
        future.add_done_callback(self.inflation_response_callback)

    def inflation_response_callback(self, future):
        try:
            self.response = future.result().occupancy_grid
            self.get_logger().info('Received inflation grid response')
            self.send_navigate_goal((14, 76))
            
        except Exception as e:
            self.get_logger().error(f'Exception while calling service: {e}')

    def send_navigate_goal(self, starting_pose):
        """ Sends a goal to the NavigateToGoal action and waits for the result or feedback condition """

        self.starting_pose = starting_pose[::-1]  # Reverse the order for the action server
        self.response_waypoint = self.response_waypoint[::-1]
        if not self.navigate_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("NavigateToGoal action server not available!")
            return

        goal_msg = NavigateToGoal.Goal()
        goal_msg.costmap = self.response  # Use the inflation grid response as the costmap
        goal_msg.start = CellCoordinateMsg(x=int(starting_pose[0]), y=int(starting_pose[1]))
        
        goal_msg.goal = CellCoordinateMsg(x=int(self.response_waypoint[0]), y=int(self.response_waypoint[1]))

        self.get_logger().info(f"Sending goal: start={starting_pose}, goal={self.response_waypoint}")

        send_goal_future = self.navigate_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        send_goal_future.add_done_callback(self.navigate_to_goal_acceptance_callback)
    
    def feedback_callback(self, feedback_msg):
        """ Process feedback from the action server """
        # pose = feedback_msg.feedback.distance_from_start
        # self.get_logger().info(f"Feedback received: Pose({pose.position.x}, {pose.position.y})")

        # # Stop if the pose is within 1.0 of the starting pose
        # if abs(pose.position.x - self.starting_pose[0]) <= 1.0 and abs(pose.position.y - self.starting_pose[1]) <= 1.0:
        #     self.get_logger().info("Robot is within 1.0 of starting position, stopping...")
        #     self.navigate_client.cancel_goal_async(self.goal_handle)

    def navigate_to_goal_acceptance_callback(self, future):
        try:
            goal_handle = future.result()
            
            if not goal_handle.accepted:
                self.get_logger().error('navigate_to_goal action goal rejected')
                self.processing_navigation = False
                return
                
            self.get_logger().info('navigate_to_goal action goal accepted')
            future = goal_handle.get_result_async()

            future.add_done_callback(self.navigate_to_goal_result_callback)
        except Exception as e:
            self.get_logger().error(f'Exception in navigate_to_goal_acceptance_callback: {e}')
            self.processing_navigation = False

    def navigate_to_goal_result_callback(self, future):
        try:
            navigation_result = future.result().result.success
            if navigation_result:
                self.get_logger().info('Navigation to goal succeeded, continuing to next goal')
            else:
                self.get_logger().info('Navigation to goal failed, retrying')
        except Exception as e:
            self.get_logger().error(f'Exception in navigate_to_goal_result_callback: {e}')

        self.processing_navigation = False

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