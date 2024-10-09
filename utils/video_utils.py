import cv2

def read_video(video_path):
    frames=[]
    cap=cv2.VideoCapture(video_path)
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()



def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def find_best_pitch_frame(transformer, video_frames):
    best_frame = None
    best_score = 0

    for frame in video_frames:
        pitch_detected, score = transformer.detect_pitch(frame)
        if pitch_detected is not None and score > best_score:
            best_frame = frame
            best_score = score

    return best_frame