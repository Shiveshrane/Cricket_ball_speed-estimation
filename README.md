# Cricket Ball Speed Estimation
This project utilizes computer vision techniques to estimate the speed of a cricket ball from video footage. It employs two YOLO (You Only Look Once) models for object detection and combines the results with geometric transformations to calculate ball speed.
Technologies Used

- Python
- YOLOv5
- Ultralytics

## Project Overview
This project aims to estimate the speed of a cricket ball using video analysis. It employs two main components:

- Ball Tracking: A fine-tuned YOLO model detects and tracks the cricket ball in each frame of the video.
- Pitch Detection: Another YOLO model identifies the cricket pitch in the video.

The data from these two models is then combined to perform view transformation and speed calculations.


## Detailed Description
The cricket ball speed estimation system works through several key steps:

Video Input: The system takes a video of a cricket match or practice session as input.
Ball Detection and Tracking:

A YOLOv5 model, fine-tuned on cricket ball images, processes each frame of the video.
It identifies the location of the ball in each frame, providing bounding box coordinates.
These coordinates are tracked across consecutive frames to establish the ball's trajectory.


### Pitch Detection:

A separate YOLOv5 model, trained to recognize cricket pitches, analyzes the video.
It provides bounding box information for the pitch in each frame.
This information is crucial for establishing a reference point and scale in the video.


### View Transformation:

Using the pitch bounding box as a reference, a perspective transformation is applied.
This step corrects for camera angle and position, providing a standardized view of the ball's path.


### Speed Calculation:

The transformed ball positions are used to calculate the ball's displacement over time.
Knowing the frame rate of the video and the real-world dimensions of a cricket pitch, the system can convert pixel distances to real-world distances.
Speed is then calculated as distance over time.

## Setup and Usage

### Environment Setup:
```
git clone https://github.com/Shiveshrane/Cricket_ball_speed-estimation.git
cd Cricket_ball_speed_estimation
pip install -r requirements.txt ```

### Run the Script:
```python main.py --video path to your video```





