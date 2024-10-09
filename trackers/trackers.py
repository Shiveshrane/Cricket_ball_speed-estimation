
from ultralytics import YOLO
import os
import supervision as sv
import cv2
import sys
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_center_bbox, get_foot_positions

class Tracker:
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i:i+batch_size], conf=0.03)
        return detections
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_obj_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            'Ball': [],
            'wickets': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            print(cls_names)
            cls_names_inv = {v: k for k, v in cls_names.items()}

            for key, value in cls_names.items():
                if value=='cricket-ball':
                    cls_names[key]='Ball'

                if value=='Ball':
                    cls_names_inv['Ball']=1              
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['Ball'].append({})
            tracks['wickets'].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if 'wickets' in cls_names_inv and cls_id == cls_names_inv['wickets']:
                    tracks['wickets'][frame_num][track_id] = {'bbox': bbox}

            for frame_detections in detection_supervision:
                bbox = frame_detections[0].tolist()
                cls_id = frame_detections[3]

                if cls_id == cls_names_inv['Ball']:
                    tracks['Ball'][frame_num][1] = {'bbox': bbox}
        return tracks

    def print_ball_path(self, tracks):
        print("Ball Path:")
        for frame_num, frame_tracks in enumerate(tracks['Ball']):
            if 1 in frame_tracks:
                bbox = frame_tracks[1]['bbox']
                center = get_center_bbox(bbox)
                print(f"Frame {frame_num}: Center Position {center}")
            else:
                print(f"Frame {frame_num}: Ball not detected")

    def add_position_to_tracks(self, tracks):
        for obj_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track in frame_tracks.items():
                    bbox = track['bbox']
                    if obj_type == 'Ball' or obj_type=='cricket-ball':
                        position = get_center_bbox(bbox)
                    elif obj_type == 'wickets':
                        position = get_foot_positions(bbox)
                    else:
                        pass
                    tracks[obj_type][frame_num][track_id]['position'] = position

    def draw_bounding_boxes(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if track_id is not None:
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_frames = []
        ball_positions = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num < len(tracks['Ball']):
                ball_dict = tracks['Ball'][frame_num]
            else:
                ball_dict = {}

            if frame_num < len(tracks['wickets']):
                wicket_dict = tracks['wickets'][frame_num]
            else:
                wicket_dict = {}
            for _, ball in ball_dict.items():
                bbox = ball['bbox']
                center = get_center_bbox(bbox)
                ball_positions.append(center)
                frame = self.draw_bounding_boxes(frame, bbox, (0, 255, 0))

            for _, wicket in wicket_dict.items():
                frame = self.draw_bounding_boxes(frame, wicket['bbox'], (0, 0, 255))

            for i in range(1, len(ball_positions)):
                cv2.line(frame, ball_positions[i-1], ball_positions[i], (0, 255, 0), 2)

            output_frames.append(frame)
        if ball_positions:
            x_positions = [pos[0] for pos in ball_positions]
            y_positions = [pos[1] for pos in ball_positions]
            frame_indices = list(range(len(ball_positions)))

            interp_x = interp1d(frame_indices, x_positions, kind='cubic', fill_value='extrapolate')
            interp_y = interp1d(frame_indices, y_positions, kind='cubic', fill_value='extrapolate')

            smooth_ball_positions = [(int(interp_x(i)), int(interp_y(i))) for i in frame_indices]

            for i in range(1, len(smooth_ball_positions)):
                cv2.line(output_frames[i], smooth_ball_positions[i-1], smooth_ball_positions[i], (0, 255, 0), 2)
        return output_frames





