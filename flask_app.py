from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import os
import cv2
from utils import read_video, save_video, get_video_fps, find_best_pitch_frame
from trackers import Tracker
from view_transformer import CricketViewTransformer
import concurrent.futures
import matplotlib.pyplot as plt


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    try:
        video_frames = read_video(video_path)
        tracker = Tracker('models\\Old models\\best.pt')
        yolo_model_path = 'models\\pitch_detection\\best (1).pt'
        transformer = CricketViewTransformer(yolo_model_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_tracks = executor.submit(tracker.get_obj_tracks, video_frames)
            future_best_frame = executor.submit(find_best_pitch_frame, transformer, video_frames)
            tracks = future_tracks.result()
            best_pitch_frame = future_best_frame.result()
        
        tracker.add_position_to_tracks(tracks)

        if best_pitch_frame is None:
            return jsonify({"error": "No suitable frame found in pitch detection"}), 400
        try:
            transformer.get_perspective_transform(best_pitch_frame)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        transformer.add_transformed_position_to_tracks(tracks['Ball'])
        fps = get_video_fps(video_path)
        max_speed = transformer.calculate_ball_speed(tracks['Ball'], fps)

        if max_speed is not None:
            return jsonify({'max_speed': f"{max_speed:.2f}"})
        else:
            return jsonify({"error": "Could not calculate max speed"}), 400
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except cv2.error as e:
        return jsonify({"error": f"OpenCV error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    

@app.route('/process_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            result = process_video(file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        return result
    else:
        return jsonify({"error": "Invalid file extension"}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)