import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array
from std_msgs.msg import Header
import math
import time
from drivable_area.bev import CameraProperties, getBirdView
import paho.mqtt.client as mqtt
import json
import time

class DrivableArea(Node):
    def __init__(self):
        super().__init__('drivable_area')
        
        self.time_of_frame = time.time()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_message = self.on_message
        self.host = "192.168.191.2"
        self.port = 1884
        self.client.connect(self.host, self.port)
        print("Connected to MQTT broker")
        self.client.subscribe("zed_image", qos=0)

        self.zed = CameraProperties(64, 68.0, 101.0, 60.0)
        self.curr_pix_size = 0.0055
        self.desired_size = 0.05
        self.scale_factor = self.curr_pix_size / self.desired_size

        self.image_height = 720
        self.image_width = 1280
        self.memory_buffer = np.full((self.image_height, self.image_width), 255)

        # Create a publisher that publishes OccupancyGrid messages on the 'occupancy_grid' topic
        self.publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.client.loop_forever()

    
    def get_occupancy_grid(self, lane, pothole):
        occupancy_grid = np.zeros((self.image_height, self.image_width))

        # time_of_frame = 0
        if lane['masks'] is not None:
            if(len(lane['masks']['xy']) != 0):
                segment = lane['masks']['xy']
                segment_array = np.array([segment], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0))
                # time_of_frame = time.time()
                # print("Lane detected")

        for i in range(occupancy_grid.shape[1]):
            if np.any(occupancy_grid[-100:, i]):
                occupancy_grid[-35:, i] = 255
        
        if pothole['boxes'] is not None:
            for segment in pothole['boxes']['xyxyn']:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([[x_min*self.image_width, y_min*self.image_height], 
                                    [x_max*self.image_width, y_min*self.image_height], 
                                    [x_max*self.image_width, y_max*self.image_height], 
                                    [x_min*self.image_width, y_max*self.image_height]], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))
                # print("Pothole detected")

        buffer_area = np.sum(occupancy_grid)//255
        buffer_time = math.exp(-buffer_area/(self.image_width*self.image_height)-0.7)
        return occupancy_grid, buffer_time#, time_of_frame


    def on_message(self, client, userdata, message):
        if message.topic == "zed_image":
            print("Received message '" + str(message.payload) + "' on topic " + message.topic)
            self.listener_callback(message)
        print("Received message '" + str(message.payload) + "' on topic " + message.topic)
    
    def listener_callback(self, message):
        
        # Get message which contains JSON of the predictions of both models
        json_message = json.loads(message.payload)
        # Get the predictions of the YOLO model
        lane = json_message["lane"]
        pothole = json_message["hole"]

        occ, buffer_time = self.get_occupancy_grid(lane, pothole)


        curr_time = time.time()
        total = np.sum(occ)

        # if total == 0:
        #     if curr_time - time_of_frame < buffer_time:
        #         occ = self.memory_buffer
        #     else:
        #         occ.fill(255)
        # else:
        #     self.memory_buffer = occ

        if total == 0:
            if curr_time - self.time_of_frame < buffer_time:
                occ = self.memory_buffer
            else:
                occ.fill(255)
        else:
            switch = np.sum(np.logical_and(self.memory_buffer, np.logical_not(occ)))/(np.sum(self.memory_buffer)/255)
            if switch >= 0.8 and curr_time - self.time_of_frame < 4:
                occ = self.memory_buffer
            elif switch >= 0.8 and curr_time - self.time_of_frame < 8:
                occ.fill(255)
            else:
                self.memory_buffer = occ
                self.time_of_frame = time.time()

        

        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occ, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)


        # Create a mask to remove the area outside the drivable area
        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - 17, bottomRight[1]], [topRight[0] - 75, topRight[1]], topLeft])
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
        rob_arr = np.full((26, new_size[0]), -1, dtype=np.int8)
        rob_arr[13][77] = 2

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))
        # combined_arr = np.where(combined_arr==0, 3, combined_arr)
        # combined_arr = np.where(combined_arr==1, 0, combined_arr)
        # combined_arr = np.where(combined_arr==3, 1, combined_arr)
        # np.savetxt('occupancy_grid.txt', combined_arr, fmt='%d')
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
        grid.info.origin.position.x = 32.0
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




