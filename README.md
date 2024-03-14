Current: 
- script runs YOLOP model on saved video + creates a replication with black pixels denoting non-drivable area
- established bare bones subscriber + publisher nodes

Discuss with Chris:
- pubish rate (0.5 seconds) -- leave as a parameter 
- how to publish occupancy grid as message (combining publisher method file and drivable lane .py)
      --- nav occupancy grid message (contains a header -- timestamp etc., data -- as an array)
- size of pixels

Next Steps: 
- create custom message to publish occupancy grid, robot (x,y) position, and grid size 
- Figure out how to parse ZED input to just recieve left frame
- how to listen to Zed node (left camera)
- judge distortion of tape measure in testing images for different angles (perspective transform)

Goals: 
- convert current "replication frame" to occupancy grid via aerial transform, search for zed2i guides
- listen to ZED nodes + create lane_detection node with publishing

Notes:
- 0.05 x 0.05 m grid size for occupancy grid
- add padding to include position of robot in numpy occupancy grid
- header data of occupancy grid data: label as ????
- https://ros2-industrial-workshop.readthedocs.io/en/latest/_source/basics/ROS2-Simple-Publisher-Subscriber.html#build-and-run

  # cv-nav-integration
Goal: CV output to Nav an occupancy grid of non-drivable area for each frame. Each frame's occupancy grid are ultimately combined to form a "world occupancy grid". 

Challenges: computational efficiency +  converting local frame to global frame where the relative position of the local origin is unknown.

Resources: 
https://docs.google.com/document/d/1lL_PE1D-wrfgzJdFESzRF_aXDcSKWAU45HWH284obUQ/edit?usp=sharing

