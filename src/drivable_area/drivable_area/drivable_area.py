import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import CameraProperties from bev
from ultralytics import YOLO

model = YOLO('./utils/nov13.pt')

class DrivableArea(Node):

    def __init__(self):
        super().__init__('drivable_area')
        self.subscription = self.create_subscription(
            Image,
            'url',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        results = model.predict(cv_image)
        masks = results[0].masks.xy

        grid = np.zeros((frame.shape[0], frame.shape[1]))

        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            instance_mask = np.ones((frame.shape[0], frame.shape[1]))
            cv2.fillPoly(instance_mask, [mask], 0)
            grid = np.logical_or(grid, instance_mask)
        occupancy_grid_display = grid.astype(np.uint8) * 255
        transformed_image = getBirdView(occupancy_grid_display, ZED)
        current_pixel_size = 0.006  # current size each pixel represents in meters
        desired_pixel_size = 0.05  # desired size each pixel should represent in meters
        scale_factor = current_pixel_size / desired_pixel_size

        new_size = (int(transformed_image.shape[1] * scale_factor), int(transformed_image.shape[0] * scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_AREA)

        rob_arr = np.full((20, 161), -1, dtype=np.uint8)
        rob_arr[10][80] = 2

        combined_arr = np.concatenate((resized_image, rob_arr), axis=0)



def main(args=None):
    rclpy.init(args=args)

    node = ImageProcessingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()