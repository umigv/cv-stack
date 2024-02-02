Current: 
- script runs YOLOP model on saved video + creates a replication with black pixels denoting non-drivable area
- established bare bones subscriber + publisher nodes

Discuss with Chris:
- pubish rate (0.5 seconds)
- how to publish occupancy grid as message (combining publisher method file and drivable lane .py)

Next Steps: 
- Figure out how to parse ZED input to just recieve left frame
- how to listen to Zed node (left camera)
- judge distortion of tape measure in testing images for different angles (perspective transform)

Goals: 
- convert current "replication frame" to occupancy grid via aerial transform, search for zed2i guides
- listen to ZED nodes + create lane_detection node with publishing
