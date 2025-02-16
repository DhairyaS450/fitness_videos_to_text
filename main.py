import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hides INFO/WARNING messages

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b formed by points a and c.
    Points a, b, c are each (x, y) pairs in 2D space.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    # Adding a small epsilon to norms to avoid ZeroDivisionError
    cosine_angle = np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6)
    # Clip and compute angle
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Additional code snippet
def calculate_neck_angle(landmarks, frame_shape):
    h, w, _ = frame_shape
    def coords(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    NOSE = mp_pose.PoseLandmark.NOSE.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    # Midpoint of left and right shoulders for the "base of the neck"
    base_neck_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) * 0.5 * w
    base_neck_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) * 0.5 * h

    # Midpoint of left and right hips for a stable reference
    base_hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) * 0.5 * w
    base_hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) * 0.5 * h

    base_neck = np.array([base_neck_x, base_neck_y])
    nose = np.array(coords(NOSE))
    base_hip = np.array([base_hip_x, base_hip_y])

    return calculate_angle(base_hip, base_neck, nose)

def calculate_spine_angle(landmarks, frame_shape):
    h, w, _ = frame_shape
    def coords(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    # Midpoint of shoulders
    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) * 0.5 * w
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) * 0.5 * h
    # Midpoint of hips
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) * 0.5 * w
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) * 0.5 * h

    # For a vertical reference, create a point slightly above or below the hip point
    # or use (hip_x, hip_y - 100) as a reference for "vertical up" 
    vertical_ref = np.array([hip_x, hip_y - 100])

    shoulder_mid = np.array([shoulder_x, shoulder_y])
    hip_mid = np.array([hip_x, hip_y])

    return calculate_angle(vertical_ref, hip_mid, shoulder_mid)

def get_frame_angles(landmarks, frame_shape):
    """
    Returns a dictionary of angles for major joints on both sides of the body:
    shoulders, elbows, hips, knees, and ankles.
    """
    h, w, _ = frame_shape

    def coords(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    # Map Mediapipe's PoseLandmark indices to easy references
    # Left side
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

    # Right side
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
    RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    angles = {}

    # Shoulder angle can be computed as the angle formed by the elbow, shoulder, and hip
    angles['left_shoulder'] = calculate_angle(
        coords(LEFT_ELBOW), 
        coords(LEFT_SHOULDER), 
        coords(LEFT_HIP)
    )
    angles['right_shoulder'] = calculate_angle(
        coords(RIGHT_ELBOW), 
        coords(RIGHT_SHOULDER), 
        coords(RIGHT_HIP)
    )

    # Elbow angle: formed by the shoulder, elbow, and wrist
    angles['left_elbow'] = calculate_angle(
        coords(LEFT_SHOULDER),
        coords(LEFT_ELBOW),
        coords(LEFT_WRIST)
    )
    angles['right_elbow'] = calculate_angle(
        coords(RIGHT_SHOULDER),
        coords(RIGHT_ELBOW),
        coords(RIGHT_WRIST)
    )

    # Hip angle: formed by the shoulder, hip, and knee
    angles['left_hip'] = calculate_angle(
        coords(LEFT_SHOULDER),
        coords(LEFT_HIP),
        coords(LEFT_KNEE)
    )
    angles['right_hip'] = calculate_angle(
        coords(RIGHT_SHOULDER),
        coords(RIGHT_HIP),
        coords(RIGHT_KNEE)
    )

    # Knee angle: formed by the hip, knee, and ankle
    angles['left_knee'] = calculate_angle(
        coords(LEFT_HIP),
        coords(LEFT_KNEE),
        coords(LEFT_ANKLE)
    )
    angles['right_knee'] = calculate_angle(
        coords(RIGHT_HIP),
        coords(RIGHT_KNEE),
        coords(RIGHT_ANKLE)
    )

    # Ankle angle: formed by the knee, ankle, and a point forward of the foot
    # Mediapipe doesn't provide a direct "foot" landmark, so this is approximate.
    # For demonstration, we can treat the toe or use the "LEFT_FOOT_INDEX" if you want another anchor.
    LEFT_FOOT_INDEX = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    RIGHT_FOOT_INDEX = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value

    angles['left_ankle'] = calculate_angle(
        coords(LEFT_KNEE),
        coords(LEFT_ANKLE),
        coords(LEFT_FOOT_INDEX)
    )
    angles['right_ankle'] = calculate_angle(
        coords(RIGHT_KNEE),
        coords(RIGHT_ANKLE),
        coords(RIGHT_FOOT_INDEX)
    )

    angles['neck'] = calculate_neck_angle(landmarks, frame_shape)
    angles['spine'] = calculate_spine_angle(landmarks, frame_shape)

    return angles

def process_video(video_path):
    """
    Processes a single video, returning a list of angle dictionaries for each valid frame.
    """
    print(f"Processing video: {video_path}")
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
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")

            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    frame_angles = get_frame_angles(results.pose_landmarks.landmark, frame.shape)
                    angle_time_series.append(frame_angles)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

    cap.release()
    print(f"Completed processing video: {video_path}")
    return angle_time_series

def generate_detailed_summary(angle_time_series):
    """
    Generates a 5–10 sentence summary of the angles for each joint over the entire video.
    This includes min, max, and average angles for left/right shoulders, elbows, hips,
    knees, and ankles.
    """
    if not angle_time_series:
        return "No pose landmarks detected in this video."

    # Collect angle stats
    summary_stats = {}
    all_joints = angle_time_series[0].keys()  # e.g. left_shoulder, right_shoulder, etc.

    for joint in all_joints:
        joint_angles = [frame[joint] for frame in angle_time_series]
        summary_stats[joint] = {
            'min': float(np.min(joint_angles)),
            'max': float(np.max(joint_angles)),
            'mean': float(np.mean(joint_angles))
        }

    # Construct a multi-sentence description
    # Customize or expand this text as needed for more specificity
    sentences = []

    sentences.append(
        f"Throughout the video, the subject's left shoulder angle ranged "
        f"from {summary_stats['left_shoulder']['min']:.1f}° to {summary_stats['left_shoulder']['max']:.1f}°, "
        f"with an average of {summary_stats['left_shoulder']['mean']:.1f}°."
    )
    sentences.append(
        f"The right shoulder angle spanned {summary_stats['right_shoulder']['min']:.1f}° to "
        f"{summary_stats['right_shoulder']['max']:.1f}°, averaging {summary_stats['right_shoulder']['mean']:.1f}° overall."
    )
    sentences.append(
        f"For the elbow joints, the left elbow showed angles from {summary_stats['left_elbow']['min']:.1f}° "
        f"up to {summary_stats['left_elbow']['max']:.1f}°, with a mean of {summary_stats['left_elbow']['mean']:.1f}°, "
        f"while the right elbow ranged {summary_stats['right_elbow']['min']:.1f}° to {summary_stats['right_elbow']['max']:.1f}°."
    )
    sentences.append(
        f"Observing the hip angles, the left hip hovered between {summary_stats['left_hip']['min']:.1f}° "
        f"and {summary_stats['left_hip']['max']:.1f}°, while the right hip displayed a similar pattern "
        f"from {summary_stats['right_hip']['min']:.1f}° to {summary_stats['right_hip']['max']:.1f}°."
    )
    sentences.append(
        f"Analysis of the knees revealed that the left knee varied from "
        f"{summary_stats['left_knee']['min']:.1f}° to {summary_stats['left_knee']['max']:.1f}°, "
        f"with the right knee achieving angles of {summary_stats['right_knee']['min']:.1f}° to "
        f"{summary_stats['right_knee']['max']:.1f}°."
    )
    sentences.append(
        f"Finally, the ankles illustrated a range of motion with the left ankle spanning "
        f"{summary_stats['left_ankle']['min']:.1f}°–{summary_stats['left_ankle']['max']:.1f}°, "
        f"and the right ankle spanning {summary_stats['right_ankle']['min']:.1f}°–{summary_stats['right_ankle']['max']:.1f}°."
    )
    sentences.append(
        "These angle patterns suggest a consistent, controlled form that can be monitored "
        "in real time to correct imbalances or deviations in posture."
    )
    sentences.append(
        "From this data, automated real-time feedback can focus on any joints that exhibit "
        "excessive or insufficient angles during the movement to help maintain proper alignment."
    )

    # Combine into one final text with 5-10 sentences
    final_summary = " ".join(sentences)
    return final_summary

def main():
    videos_base_dir = "data"
    output_file = "video_summaries.txt"

    if not os.path.exists(videos_base_dir):
        print(f"Error: Directory '{videos_base_dir}' does not exist.")
        return

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
                    angle_data = process_video(video_path)
                    summary = generate_detailed_summary(angle_data)
                    out_f.write(f"File: {video_path}\nSummary: {summary}\n\n")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    out_f.write(f"File: {video_path}\nError: Failed to process video - {str(e)}\n\n")

    print(f"\nProcessing complete. Processed {video_count} videos.")

if __name__ == "__main__":
    main()
