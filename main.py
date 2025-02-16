import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This hides INFO/WARNING messages
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    # Same as before
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_frame_angles(landmarks, frame_shape):
    # Return a dictionary of angles for the entire body
    h, w, _ = frame_shape

    def coords(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    angles = {}
    # Example for left knee, left hip, left ankle
    angles['left_knee'] = calculate_angle(
        coords(mp_pose.PoseLandmark.LEFT_HIP.value),
        coords(mp_pose.PoseLandmark.LEFT_KNEE.value),
        coords(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    )
    # Add more angles for shoulders, elbows, etc.
    angles['left_elbow'] = calculate_angle(
        coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        coords(mp_pose.PoseLandmark.LEFT_ELBOW.value),
        coords(mp_pose.PoseLandmark.LEFT_WRIST.value)
    )
    # Continue for right side or other joints you care about...
    return angles

def process_video(video_path):
    print(f"Processing video: {video_path}")  # Add progress indicator
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    angle_time_series = []
    frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Show progress every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")

            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    # Calculate angles for this frame
                    frame_angles = get_frame_angles(results.pose_landmarks.landmark, frame.shape)
                    angle_time_series.append(frame_angles)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

        cap.release()
    print(f"Completed processing video: {video_path}")
    return angle_time_series

def identify_exercise(angle_time_series):
    # Simple placeholder logic (heuristic)
    # Example: if the left_knee angle range is large, guess "Squat," etc.
    if not angle_time_series:
        return "Unknown"

    left_knee_angles = [f['left_knee'] for f in angle_time_series if 'left_knee' in f]
    knee_range = np.max(left_knee_angles) - np.min(left_knee_angles)

    # If knee range is big, guess squat
    if knee_range > 70:
        return "Squat"
    # Other heuristics for different exercises...
    return "Unknown"

def generate_text_summary(exercise_type, angle_time_series):
    if not angle_time_series:
        return "No pose landmarks detected in the video."

    if exercise_type == "Squat":
        left_knee = [f['left_knee'] for f in angle_time_series if 'left_knee' in f]
        min_knee = np.min(left_knee)
        max_knee = np.max(left_knee)
        avg_knee = np.mean(left_knee)

        return (f"This video appears to show a Squat. "
                f"Knee angle ranged from {min_knee:.1f}° to {max_knee:.1f}°, averaging {avg_knee:.1f}°. "
                f"This suggests a complete movement through the squat range of motion.")

    # Add more descriptions for each recognized exercise
    return "The movement does not match a recognized exercise pattern."

if __name__ == "__main__":
    videos_base_dir = "data"
    output_file = "video_summaries.txt"

    # Check if data directory exists
    if not os.path.exists(videos_base_dir):
        print(f"Error: Directory '{videos_base_dir}' does not exist")
        exit(1)

    video_count = 0
    with open(output_file, "w") as out_f:
        for root, dirs, files in os.walk(videos_base_dir):
            mp4_files = [f for f in files if f.lower().endswith(".mp4")]
            if not mp4_files:
                print(f"No MP4 files found in {root}")
                continue
                
            print(f"Found {len(mp4_files)} MP4 files in {root}")
            
            for file in mp4_files:
                video_count += 1
                video_path = os.path.join(root, file)
                print(f"\nProcessing video {video_count}: {file}")
                
                try:
                    angles = process_video(video_path)
                    exercise_type = identify_exercise(angles)
                    summary = generate_text_summary(exercise_type, angles)
                    out_f.write(f"File: {video_path}\nSummary: {summary}\n\n")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    out_f.write(f"File: {video_path}\nError: Failed to process video - {str(e)}\n\n")

    print(f"\nProcessing complete. Processed {video_count} videos.")
