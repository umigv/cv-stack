import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("./utils/nov13.pt")

video = cv2.VideoCapture("./utils/comp23_2.mp4")


fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height), isColor=True)
out2 = cv2.VideoWriter('occupancy_grid.avi', fourcc, fps, (width, height), isColor=True)

print("Running")

try:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25)

        masks = results[0].masks.xy

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

        out.write(overlay)
        out2.write(cv2.cvtColor(occupancy_grid_display, cv2.COLOR_GRAY2BGR))

except KeyboardInterrupt:
    pass
video.release()
# cv2.destroyAllWindows()