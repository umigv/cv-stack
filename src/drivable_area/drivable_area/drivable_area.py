# THIS WILL SUBSCRIBE TO ZED CAMERA AND PRODUCE AN OCCUPANCY GRID

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# from bev import CameraProperties
from ultralytics import YOLO
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array

model = YOLO('src/drivable_area/drivable_area/utils/nov13.pt')

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


    # TODO: translate the occupancy grid into birds eye view
    # TODO: publish the occupancy grid for nav team
    def listener_callback(self, msg):
        
        #converts Image message to cv2 type
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # runs YOLO predictions on the cv2 image
        results = model.predict(frame)
        masks = results[0].masks.xy
        print(masks)
        
        grid = np.zeros((frame.shape[0], frame.shape[1]))
    
        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            instance_mask = np.ones((frame.shape[0], frame.shape[1]))
            cv2.fillPoly(instance_mask, [mask], 0)
            grid = np.logical_or(grid, instance_mask)
            

        # Convert grid to uint8 for display
        occupancy_grid_display = grid.astype(np.uint8) * 255

        # Overlay occupancy grid on the original frame
        overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occupancy_grid_display, cv2.COLOR_GRAY2BGR), 0.5, 0)


        cv2.imshow('Original Video', overlay)
        cv2.imshow('Occupancy Grid', occupancy_grid_display)
        
        if cv2.waitKey(1) == ord('q'):
            return

        # grid = np.zeros((frame.shape[0], frame.shape[1]))

        # for mask in masks:
        #     mask = np.array(mask, dtype=np.int32)
        #     instance_mask = np.ones((frame.shape[0], frame.shape[1]))
        #     cv2.fillPoly(instance_mask, [mask], 0)
        #     grid = np.logical_or(grid, instance_mask)

        # occupancy_grid_display = grid.astype(np.uint8) * 255
        # transformed_image = getBirdView(occupancy_grid_display, ZED)
        # current_pixel_size = 0.006  # current size each pixel represents in meters
        # desired_pixel_size = 0.05  # desired size each pixel should represent in meters
        # scale_factor = current_pixel_size / desired_pixel_size

        # new_size = (int(transformed_image.shape[1] * scale_factor), int(transformed_image.shape[0] * scale_factor))
        # resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_AREA)

        # rob_arr = np.full((20, 161), -1, dtype=np.uint8)
        
        # rob_arr[10][80] = 2

        # combined_arr = np.concatenate((resized_image, rob_arr), axis=0)
        # grid_to_publish = numpy_to_occupancy_grid(combined_arr)
        # print(grid_to_publish)
        
        # out.write(overlay)
        # out2.write(cv2.cvtColor(occupancy_grid_display, cv2.COLOR_GRAY2BGR))


def main(args=None):
    rclpy.init(args=args)

    node = DrivableArea()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

# def numpy_to_occupancy_grid(arr, info=None, x=80,y=77):
#     if not len(arr.shape) == 2:
#         raise TypeError('Array must be 2D')
#     if not arr.dtype == np.int8:
#         raise TypeError('Array must be of int8s')

#     grid = OccupancyGrid()
#     if isinstance(arr, np.ma.MaskedArray):
#         # We assume that the masked value are already -1, for speed
#         arr = arr.data

#     grid.data = Array('b', arr.ravel().astype(np.int8))
#     grid.info = info or MapMetaData()
#     grid.info.height = arr.shape[0]
#     grid.info.width = arr.shape[1]
#     grid.info.geometry_msgs/Pose origin = (x,y,0)

#     return grid
if __name__ == '__main__':
    main()