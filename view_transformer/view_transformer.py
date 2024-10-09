import numpy as np
import cv2
from ultralytics import YOLO

class CricketViewTransformer:
    def __init__(self, yolo_model_path):
        self.pitch_length = 20.12  
        self.pitch_width = 3.05 
        self.model = YOLO(yolo_model_path)

    def detect_pitch(self, image):
        results = self.model.predict(image, conf=0.03)
        if len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes) > 0:
                box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
                if len(box) >= 4:
                    return box[:4], 1 
        return None, 0  

    def get_perspective_transform(self, image):
        pitch_box, score = self.detect_pitch(image)
        if pitch_box is None or len(pitch_box) < 4:
            print("No pitch detected in the image or invalid pitch box")
            raise ValueError("No pitch detected in the image or invalid pitch box")
        x1, y1, x2, y2 = pitch_box
        self.pixel_vertices = np.array([
            [x1, y2],  
            [x1, y1],  
            [x2, y1],  
            [x2, y2]   
        ], dtype=np.float32)

        self.target_vertices = np.array([
            [0, self.pitch_width],
            [0, 0],
            [self.pitch_length, 0],
            [self.pitch_length, self.pitch_width]
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for frame_num, frame_tracks in enumerate(tracks):
            for track_id, track in frame_tracks.items():
                if 'position' in track:
                    position = track['position']
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                        tracks[frame_num][track_id]['position_adjusted'] = position_transformed
                    else:
                        tracks[frame_num][track_id]['position_adjusted'] = None


    def calculate_ball_speed(self, tracks, fps):
        speeds = []
        for i in range(1, len(tracks)):
            prev_frame = tracks[i-1]
            curr_frame = tracks[i]
            
            for track_id in curr_frame:
                if track_id in prev_frame:
                    prev_pos = prev_frame[track_id].get('position_adjusted')
                    curr_pos = curr_frame[track_id].get('position_adjusted')
                    
                    if prev_pos is not None and curr_pos is not None:
                        prev_pos = np.array(prev_pos)
                        curr_pos = np.array(curr_pos)
                        distance = np.linalg.norm(curr_pos - prev_pos)
                        time = 1 / fps  
                        speed = distance / time 
                        #speed in km/hr
                        speed = speed * 3.6
                        speeds.append(speed)
        
        if speeds:
            return max(speeds)
        else:
            return 0