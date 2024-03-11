import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class TestSubscriber(Node):

    def __init__(self):
        super().__init__('test_subscriber')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('Received occupancy grid')
        self.get_logger().info('Header: %s' % msg.header)
        self.get_logger().info('Info: %s' % msg.info)
        data = msg.data
        width = msg.info.width
        height = msg.info.height
        arr = np.array(data).reshape((height, width))
        self.get_logger().info('Numpy array shape: %s' % str(arr.shape))


def main(args=None):
    rclpy.init(args=args)

    node = TestSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()