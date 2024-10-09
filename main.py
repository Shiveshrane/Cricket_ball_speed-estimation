from utils import read_video, save_video, get_video_fps
from trackers import Tracker
from view_transformer import CricketViewTransformer
import numpy as np
import cv2
import argparse

def find_best_pitch_frame(transformer, video_frames):
    best_frame = None
    best_score = 0

    for frame in video_frames:
        pitch_detected, score = transformer.detect_pitch(frame)
        if pitch_detected is not None and score > best_score:
            best_frame = frame
            best_score = score

    return best_frame

def main(video_path):
    video_frames = read_video(video_path) 
    if not video_frames:
        print(f"No frames read from video at path: {video_path}")
        return
    tracker = Tracker('models\\Old models\\best.pt')
    tracks = tracker.get_obj_tracks(video_frames)
    tracker.add_position_to_tracks(tracks)
    yolo_model_path = 'models\\pitch_detection\\best (1).pt'
    transformer = CricketViewTransformer(yolo_model_path)
    best_pitch_frame = find_best_pitch_frame(transformer, video_frames)
    if best_pitch_frame is None:
        print("No suitable frame found for pitch detection. Using the first frame as fallback.")
        best_pitch_frame = video_frames[0]
    try:
        transformer.get_perspective_transform(best_pitch_frame)
    except ValueError as e:
        print(e)
        return
    transformer.add_transformed_position_to_tracks(tracks['Ball'])
    output_frames = tracker.draw_annotations(video_frames, tracks)
    fps = get_video_fps(video_path)
    max_speed = transformer.calculate_ball_speed(tracks['Ball'], fps)
    
    for frame_num, frame_tracks in enumerate(tracks['Ball']):
        for track_id in frame_tracks:
            if frame_num < len(tracks['Ball']):
                tracks['Ball'][frame_num][track_id]['speed'] = max_speed
            else:
                tracks['Ball'][frame_num][track_id]['speed'] = None
    if max_speed is not None:
        print(f"Max ball speed: {max_speed:.2f} km/hr")
    else:
        print("Max ball speed: No valid speeds calculated")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cricket Ball Speed Estimation")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    args = parser.parse_args()
    main(args.video)
