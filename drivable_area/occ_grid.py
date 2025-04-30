import cv2
import numpy as np
import os
import json
from ultralytics import YOLO

class occ_grid:
    def __init__(self, video_path, yolo_path = None):
        self.hsv_image = None
        self.hsv_filters = {}  # Map of filter names to HSV bounds
        self.setup = False
        self.image = None
        self.gamma_image = None
        if yolo_path is not None:
            self.laneline_model = YOLO(yolo_path)
        else:
            self.laneline_model = None
        self.laneline_mask = None
        self.final = None
        self.video_path = video_path
        self.load_hsv_values()
        
        
    def load_hsv_values(self):
        if os.path.exists('utils/hsv_values.json'):
            with open('utils/hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
                self.hsv_filters = all_hsv_values.get(str(self.video_path), {})
        else:
            # Initialize with an empty filter map if the JSON file doesn't exist
            self.hsv_filters = {}

    def save_hsv_values(self):
        all_hsv_values = {}
        if os.path.exists('utils/hsv_values.json'):
            with open('utils/hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)
        all_hsv_values[str(self.video_path)] = self.hsv_filters
        with open('utils/hsv_values.json', 'w') as file:
            json.dump(all_hsv_values, file, indent=4)
        
    def on_button_click(self, value):
        if(value == 1):
            self.setup = False
            
    def __update_filter(self, filter_name, key, value):
        self.hsv_filters[filter_name][key] = value
        _, filters = self.update_mask()
        cv2.imshow("Mask", filters[filter_name])

    def clear_filter(self, filter_name):
        if os.path.exists('utils/hsv_values.json'):
            with open('utils/hsv_values.json', 'r') as file:
                all_hsv_values = json.load(file)

            if self.video_path in all_hsv_values:
                if filter_name in all_hsv_values[self.video_path]:
                    del all_hsv_values[self.video_path][filter_name]

                    if not all_hsv_values[self.video_path]:
                        del all_hsv_values[self.video_path]

                    with open('utils/hsv_values.json', 'w') as file:
                        json.dump(all_hsv_values, file, indent=4)
                    print(f"Filter '{filter_name}' cleared for video '{self.video_path}'.")
                else:
                    print(f"Filter '{filter_name}' does not exist for video '{self.video_path}'.")
            else:
                print(f"Video '{self.video_path}' does not exist in the JSON file.")
        else:
            print("No HSV values file found.")
                
    def adjust_gamma(self, gamma=0.4):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.gamma_image = cv2.LUT(self.image, table)

    def get_lane_lines_YOLO(self):
        # Get the driveable area of one frame and return the inverted mask
        results = self.laneline_model.predict(self.image, conf=0.7)[0]
        self.laneline_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if(results.masks is not None):
            segment = results.masks.xy[0]
            segment_array = np.array([segment], dtype=np.int32)
            cv2.fillPoly(self.laneline_mask, [segment_array], color=(255, 0, 0))
        
    def tune_hsv(self, filter_name):
        # if force is True and 
        if filter_name not in self.hsv_filters:
            # Initialize default values for the new filter 
            self.hsv_filters[filter_name] = {
                'h_upper': 179, 'h_lower': 0,
                's_upper': 255, 's_lower': 0,
                'v_upper': 255, 'v_lower': 0
            }
        
        filter_values = self.hsv_filters[filter_name]
        self.setup = True
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        cv2.namedWindow('control pannel')
        cv2.createTrackbar('H_upper', 'control pannel', filter_values['h_upper'], 179,
                           lambda v: self.__update_filter(filter_name, 'h_upper', v))
        cv2.createTrackbar('H_lower', 'control pannel', filter_values['h_lower'], 179,
                           lambda v: self.__update_filter(filter_name, 'h_lower', v))
        cv2.createTrackbar('S_upper', 'control pannel', filter_values['s_upper'], 255,
                           lambda v: self.__update_filter(filter_name, 's_upper', v))
        cv2.createTrackbar('S_lower', 'control pannel', filter_values['s_lower'], 255,
                           lambda v: self.__update_filter(filter_name, 's_lower', v))
        cv2.createTrackbar('V_upper', 'control pannel', filter_values['v_upper'], 255,
                           lambda v: self.__update_filter(filter_name, 'v_upper', v))
        cv2.createTrackbar('V_lower', 'control pannel', filter_values['v_lower'], 255,
                           lambda v: self.__update_filter(filter_name, 'v_lower', v))
        cv2.createTrackbar('Done Tuning', 'control pannel', 0, 1, self.on_button_click)

        while self.setup:
            ret, frame = cap.read()
            if not ret:
                # If the video ends, reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask, dict = self.update_mask()

            cv2.imshow('Video', frame)
            cv2.imshow('Mask', dict[filter_name])

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'Esc' to exit the loop
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_hsv_values()
        
    def update_mask(self):
        combined_mask = None
        masks = {}

        for filter_name, bounds in self.hsv_filters.items():
            lower_bound = np.array([bounds["h_lower"], bounds['s_lower'], bounds['v_lower']])
            upper_bound = np.array([bounds['h_upper'], bounds['s_upper'], bounds['v_upper']])
            mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 200 # Adjust based on noise size
            final = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    cv2.drawContours(final, [cnt], -1, 255, thickness=cv2.FILLED)

            # Combine masks
            if combined_mask is None:
                combined_mask = final
            else:
                combined_mask = cv2.bitwise_or(combined_mask, final)

            masks[filter_name] = final

        return combined_mask, masks
    
    
    def find_line_cols(self, row, occupancy_grid):
        line1 = 0
        in_a_row = 0
        for col in range(occupancy_grid.shape[1]):
            if (occupancy_grid[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line1 = col
        line2 = self.image_width-10
        in_a_row = 0
        for col in range(line1 + 200, occupancy_grid.shape[1]):
            if (occupancy_grid[row, col] > 0):
                in_a_row += 1
                if(in_a_row == 5):
                    line2 = col
        return line1, line2
                
    
    def find_closest_row(self, occupancy_grid):
        # Iterate from the bottom to the top
        in_a_row = 0
        for row in range(occupancy_grid.shape[0] - 1, -1, -1):
            if np.any(occupancy_grid[row, :] > 0):
                in_a_row += 1
                if(in_a_row == 7):
                    return row-50
            else:
                in_a_row = 0
        return None
    
    def nomansland_func(self, combined, dict):
        occupancy_grid = dict['white']
        closest_row = self.find_closest_row(occupancy_grid)
        if closest_row is not None:
            l1, l2 = self.find_line_cols(closest_row, occupancy_grid=occupancy_grid)
            cv2.line(occupancy_grid, (l1, closest_row), (0, occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
            cv2.line(occupancy_grid, (l2, closest_row), (occupancy_grid.shape[1], occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
            cv2.line(combined, (l1, closest_row), (0, occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-left corner
            cv2.line(combined, (l2, closest_row), (occupancy_grid.shape[1], occupancy_grid.shape[0]), (255, 0, 0), 10)  # Line to bottom-right corner
    

    def get_mask(self, frame, yolo=False, nomandsland=False):
        self.image = frame
        self.adjust_gamma()
        self.hsv_image = cv2.cvtColor(self.gamma_image, cv2.COLOR_BGR2HSV)
        #if yolo then combine hsv and yolo output, if not yolo then just return hsv mask
        if nomandsland:
            #only use the white mask and 
            combined, dict = self.update_mask()
            self.nomansland_func(combined, dict)
            return combined, dict
        if yolo:
            self.get_lane_lines_YOLO()
            mask, dict = self.update_mask()
            mask = cv2.bitwise_or(mask, self.laneline_mask)
            dict['yolo'] = self.laneline_mask
            if(dict.find('white') != -1):
                dict['white'] = cv2.bitwise_or(dict['white'], self.laneline_mask)
            return mask, dict
        else:
            return self.update_mask()
