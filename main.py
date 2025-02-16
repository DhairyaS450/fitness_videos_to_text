import cv2
import mediapipe as mp
import numpy as np
import os  # NEW: Import os to walk directories

# Initialize MediaPipe pose components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Utility function:
    Calculates the angle (in degrees) between three points (a, b, c).
    Each point is a [x, y] coordinate.
    """
    a = np.array(a)  # First coordinate
    b = np.array(b)  # Midpoint (joint)
    c = np.array(c)  # Last coordinate

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Lists to store angle data, etc.
    knee_angles = []  # Example: tracking knee angle to detect squats

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Pose
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Helper function to get a landmark's (x, y) coords in image space
                def get_coord(idx):
                    h, w, _ = frame.shape
                    return [landmarks[idx].x * w, landmarks[idx].y * h]

                # Example: let's calculate angle of left knee
                left_hip = get_coord(mp_pose.PoseLandmark.LEFT_HIP.value)
                left_knee = get_coord(mp_pose.PoseLandmark.LEFT_KNEE.value)
                left_ankle = get_coord(mp_pose.PoseLandmark.LEFT_ANKLE.value)

                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                knee_angles.append(knee_angle)

            # If you want to visualize in real-time, uncomment these lines:
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release resources
        cap.release()
        # cv2.destroyAllWindows()

    return knee_angles

def generate_text_summary(knee_angles):
    """
    Simple rule-based approach for demonstration:
    - If the knee angle repeatedly transitions from ~40° to ~150°, we might guess 'squats'.
    - Provide a rough textual summary based on these transitions.
    """
    if not knee_angles:
        return "No keypoints or movement data detected."

    # Basic threshold to decide if it looks like squat form
    angle_min = np.min(knee_angles)
    angle_max = np.max(knee_angles)

    # Arbitrary thresholds for demonstration
    if angle_min < 60 and angle_max > 120:
        description = (f"This video likely demonstrates a squat movement, "
                       f"with knee angles varying from around {angle_min:.1f}° to {angle_max:.1f}°.")
    else:
        description = (f"Movements do not match a typical squat range. Knee angles: "
                       f"min {angle_min:.1f}°, max {angle_max:.1f}°.")

    return description

if __name__ == "__main__":
    # Base directory where videos are stored
    videos_base_dir = "data"

    # Output file to store summaries
    output_file = "video_summaries.txt"

    with open(output_file, "w") as out_f:
        # Walk through the videos base directory recursively
        for root, dirs, files in os.walk(videos_base_dir):
            for file in files:
                if file.lower().endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    print("Processing:", video_path)
                    angles = process_video(video_path)
                    summary = generate_text_summary(angles)
                    out_f.write(f"File: {video_path}\n")
                    out_f.write(f"Summary: {summary}\n\n")
                    print("Summary:", summary)
