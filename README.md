Current: 
- script runs YOLOP model on saved video + creates a replication with black pixels denoting non-drivable area
- established bare bones subscriber + publisher nodes

Discuss with Chris:
- pubish rate (0.5 seconds) -- leave as a parameter 
- how to publish occupancy grid as message (combining publisher method file and drivable lane .py)
      --- nav occupancy grid message (contains a header -- timestamp etc., data -- as an array)
- size of pixels

Next Steps: 
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
