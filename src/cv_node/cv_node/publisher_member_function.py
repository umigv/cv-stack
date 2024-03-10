#initialize all first: publisher and subsriber, globals
    #read ZED(camera) stream, pass ZED frame to YOLO model, that returns a numpy array occupancy grid, so we process the drivable area numpy: this includes the birds eye view transform,
    #replace unkown with -1's, append robot location,          convert numpy to occupancy grid(the function that i put in), publish
    #happens every 0.5 seconds the whole loop

   # 3 has functions for each part, the rest should be short, one line ish    all that in main function 
   #how to pass the occupancy grid through the spin function with the timer i need to do the last part, the publishing



#My Implementation
#This code should create a ROS node that publishes an OccupancyGrid message every 0.5 seconds. 
#The occupancy grid is obtained by converting a numpy array, 
#which should be implemented in the get_occupancy_grid method.

# Import necessary modules
import rclpy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


# Define a class MinimalPublisher that inherits from Node
class MinimalPublisher(Node):

    def __init__(self):
        # Call the constructor of the parent class Node
        super().__init__('minimal_publisher')
        # Create a publisher that publishes OccupancyGrid messages on the 'topic' topic
        self.publisher_ = self.create_publisher(OccupancyGrid, 'topic', 10)
        # Create a timer that calls the timer_callback method every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Get a new occupancy grid
        grid = self.get_occupancy_grid()
        # Publish the occupancy grid
        self.publisher_.publish(grid)
        # Log a message
        self.get_logger().info('Publishing occupancy grid')

    def get_occupancy_grid(self):
        # This method is to return a new occupancy grid
        # This is a placeholder implementation, actual will be added later
        # Create a dummy 2D numpy array of type np.int8
        arr = np.array([[0, 0,  0, 0, 1]], dtype=np.int8)

                # Convert the numpy array to an occupancy grid
        #grid = numpy_to_occupancy_grid(arr)
        grid = OccupancyGrid()
        grid.data = arr
        
        
        return grid

# This function converts a numpy array to an occupancy grid
def numpy_to_occupancy_grid(arr= np.array([[0, 0], [0,1], [0, 1]], dtype=np.int8), info=None):
    # Check if the array is 2D and of type int8
    
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')
    if not arr.dtype == np.int8:
        raise TypeError('Array must be of int8s')

    # Create an OccupancyGrid object
    grid = OccupancyGrid()
    if isinstance(arr, np.ma.MaskedArray):
        # If the array is a masked array, use the underlying data
        arr = arr.data

    # Set the data and info of the grid
    grid.data = arr
    grid.info = info or MapMetaData()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]

    return grid

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



# Create a dummy 2D numpy array of type np.int8
arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)

# Convert the numpy array to an occupancy grid
grid = numpy_to_occupancy_grid(arr)

# Print the data and info of the grid to check if the conversion was successful
print(grid.data)
print(grid.info)