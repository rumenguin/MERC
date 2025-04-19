import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Init MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Set base paths
video_base = "/Users/rumenguin/Research/MERC/EmoReact/Data"
output_base = "/Users/rumenguin/Research/MERC/EmoReact/Body_features"
splits = {"Train": "Train_feat", "Test": "Test_feat", "Validation": "Val_feat"}

# Create output dirs
for folder in splits.values():
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# Process videos
for split, out_dir in splits.items():
    input_dir = os.path.join(video_base, split)
    output_dir = os.path.join(output_base, out_dir)
    
    for video_file in tqdm(os.listdir(input_dir), desc=f"Processing {split}"):
        if not video_file.endswith(".mp4"):
            continue
            
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            if results.pose_landmarks:
                landmarks = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]
                all_landmarks.append((frame_count, landmarks))
                frame_count += 1
                
        cap.release()
        
        # Ensure we have landmarks before saving
        if all_landmarks:
            # Get frames and landmarks
            frame_numbers = [item[0] for item in all_landmarks]
            landmarks_only = [item[1] for item in all_landmarks]
            landmarks_array = np.array(landmarks_only, dtype=np.float32)
            
            # Save features as npy
            output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.npy")
            np.save(output_path, landmarks_array)
            
            # Save features as txt (comma-separated)
            txt_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.txt")
            
            # Generate column names
            # First columns similar to OpenFace format
            first_columns = ["frame", "timestamp", "confidence", "success"]
            
            # MediaPipe specific columns
            landmark_names = [
                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
                'right_eye_inner', 'right_eye', 'right_eye_outer', 
                'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 
                'left_index', 'right_index', 'left_thumb', 'right_thumb', 
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 
                'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 
                'left_foot_index', 'right_foot_index'
            ]
            
            pose_columns = []
            for name in landmark_names:
                pose_columns.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"])
            
            # Combine all column names
            all_columns = first_columns + pose_columns
            header = ','.join(all_columns)
            
            # Write the data in OpenFace-style format
            with open(txt_output_path, 'w') as f:
                # Write header
                f.write(f"{header}\n")
                
                # Write data rows
                for i, frame_data in enumerate(all_landmarks):
                    frame_num = frame_data[0]
                    frame_landmarks = frame_data[1]
                    
                    # Get average confidence from landmark visibilities
                    avg_confidence = np.mean([lm[3] for lm in frame_landmarks])
                    
                    # Flatten landmarks
                    flat_landmarks = [val for lm in frame_landmarks for val in lm]
                    
                    # Create row with format: frame, timestamp, confidence, success, [landmarks...]
                    row_data = [
                        frame_num,                   # frame number
                        frame_num / 30.0,            # timestamp (assuming 30fps)
                        f"{avg_confidence:.6f}",     # confidence
                        1                            # success (1=yes)
                    ] + flat_landmarks
                    
                    row_str = ','.join(map(str, row_data))
                    f.write(f"{row_str}\n")

# Detailed Mediapipe Pose (Not Required as already has OpenFace Facial Expressions)
'''
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import math
import time

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create detector instances
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Set base paths
video_base = "/Users/rumenguin/Research/MERC/EmoReact/Data"
output_base = "/Users/rumenguin/Research/MERC/EmoReact/Body_features"
splits = {"Train": "Train_feat", "Test": "Test_feat", "Validation": "Val_feat"}

# Create output dirs
for folder in splits.values():
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# Calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    a: first point [x,y,z]
    b: middle point [x,y,z]
    c: last point [x,y,z]
    Returns: angle in degrees
    """
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Calculate velocity between frames
def calculate_velocity(curr_landmarks, prev_landmarks):
    """
    Calculate the velocity of landmarks between frames
    """
    if prev_landmarks is None:
        return np.zeros(len(curr_landmarks))
    
    velocities = []
    for i in range(len(curr_landmarks)):
        curr = np.array(curr_landmarks[i][:3])
        prev = np.array(prev_landmarks[i][:3])
        velocity = np.linalg.norm(curr - prev)
        velocities.append(velocity)

    return velocities

# Get posture characteristics
def get_posture_characteristics(landmarks):
    """
    Extract posture characteristics from landmarks
    """
    if landmarks is None or len(landmarks) < 33:  # Full pose has 33 landmarks
        return {}
    
    # Key landmark indices
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    
    # Calculate torso angle (head to hips)
    mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                    (left_shoulder[1] + right_shoulder[1])/2,
                    (left_shoulder[2] + right_shoulder[2])/2, 1.0]
    
    mid_hip = [(left_hip[0] + right_hip[0])/2, 
               (left_hip[1] + right_hip[1])/2,
               (left_hip[2] + right_hip[2])/2, 1.0]
    
    # Calculate vertical axis
    vertical = [mid_hip[0], mid_hip[1] - 1, mid_hip[2], 1.0]
    
    # Calculate torso forward angle
    torso_angle = calculate_angle(vertical, mid_hip, mid_shoulder)
    
    # Calculate shoulder symmetry (angle between shoulders)
    shoulder_angle = calculate_angle(left_shoulder, mid_shoulder, right_shoulder)
    
    # Calculate hip symmetry (angle between hips)
    hip_angle = calculate_angle(left_hip, mid_hip, right_hip)
    
    # Calculate knee bend angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    # Calculate shoulder width and hip width
    shoulder_width = np.linalg.norm(np.array(left_shoulder[:3]) - np.array(right_shoulder[:3]))
    hip_width = np.linalg.norm(np.array(left_hip[:3]) - np.array(right_hip[:3]))
    
    # Posture ratio (shoulder width / hip width)
    posture_ratio = shoulder_width / hip_width if hip_width > 0 else 0
    
    return {
        "torso_angle": torso_angle,
        "shoulder_symmetry": shoulder_angle,
        "hip_symmetry": hip_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "posture_ratio": posture_ratio
    }

# Extract facial emotion cues
def extract_facial_cues(face_landmarks):
    """
    Extract facial emotion cues from landmarks
    """
    if face_landmarks is None:
        return {}
    
    # Convert to numpy array for easier manipulation
    points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
    
    # Indices for key facial features
    left_eye = [33, 133]  # Left eye corners
    right_eye = [362, 263]  # Right eye corners
    left_eyebrow = [65, 105]  # Left eyebrow points
    right_eyebrow = [295, 334]  # Right eyebrow points
    mouth_corners = [61, 291]  # Mouth corners
    top_lip = 13  # Top lip
    bottom_lip = 14  # Bottom lip
    
    # Eye openness (vertical distance / horizontal distance)
    left_eye_open = np.linalg.norm(points[160] - points[144]) / np.linalg.norm(points[left_eye[0]] - points[left_eye[1]])
    right_eye_open = np.linalg.norm(points[385] - points[374]) / np.linalg.norm(points[right_eye[0]] - points[right_eye[1]])
    
    # Eyebrow raise (distance from eye to eyebrow)
    left_eyebrow_raise = np.linalg.norm(points[left_eyebrow[0]] - points[left_eye[0]])
    right_eyebrow_raise = np.linalg.norm(points[right_eyebrow[0]] - points[right_eye[0]])
    
    # Mouth openness
    mouth_open = np.linalg.norm(points[top_lip] - points[bottom_lip])
    
    # Mouth width
    mouth_width = np.linalg.norm(points[mouth_corners[0]] - points[mouth_corners[1]])
    
    # Smile indicator (mouth corner height relative to center)
    mouth_center = (points[mouth_corners[0]] + points[mouth_corners[1]]) / 2
    mouth_corner_height = (points[mouth_corners[0]][1] + points[mouth_corners[1]][1]) / 2 - mouth_center[1]
    
    return {
        "left_eye_openness": float(left_eye_open),
        "right_eye_openness": float(right_eye_open),
        "left_eyebrow_raise": float(left_eyebrow_raise),
        "right_eyebrow_raise": float(right_eyebrow_raise),
        "mouth_openness": float(mouth_open),
        "mouth_width": float(mouth_width),
        "smile_indicator": float(mouth_corner_height)
    }

# Process videos
for split, out_dir in splits.items():
    input_dir = os.path.join(video_base, split)
    output_dir = os.path.join(output_base, out_dir)
    
    for video_file in tqdm(os.listdir(input_dir), desc=f"Processing {split}"):
        if not video_file.endswith(".mp4"):
            continue
            
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Data storage
        all_frame_data = {
            "pose_landmarks": [],
            "face_landmarks": [],
            "left_hand_landmarks": [],
            "right_hand_landmarks": [],
            "velocities": [],
            "posture": [],
            "facial_cues": [],
            "frame_timestamps": []
        }
        
        frame_count = 0
        prev_pose_landmarks = None
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_time = frame_count / fps
            frame_count += 1
            
            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            pose_results = pose.process(image)
            face_results = face_mesh.process(image)
            hands_results = hands.process(image)
            
            # Extract pose landmarks
            if pose_results.pose_landmarks:
                pose_lm = [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark]
                all_frame_data["pose_landmarks"].append(pose_lm)
                
                # Calculate velocities
                velocities = calculate_velocity(pose_lm, prev_pose_landmarks)
                all_frame_data["velocities"].append(velocities)
                
                # Extract posture characteristics
                posture = get_posture_characteristics(pose_lm)
                all_frame_data["posture"].append(posture)
                
                prev_pose_landmarks = pose_lm
            else:
                all_frame_data["pose_landmarks"].append(None)
                all_frame_data["velocities"].append(None)
                all_frame_data["posture"].append(None)
            
            # Extract face landmarks
            if face_results.multi_face_landmarks:
                face_lm = [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark]
                all_frame_data["face_landmarks"].append(face_lm)
                
                # Extract facial emotional cues
                facial_cues = extract_facial_cues(face_results.multi_face_landmarks[0])
                all_frame_data["facial_cues"].append(facial_cues)
            else:
                all_frame_data["face_landmarks"].append(None)
                all_frame_data["facial_cues"].append(None)
            
            # Extract hand landmarks
            left_hand = None
            right_hand = None
            
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    if hand_idx >= len(hands_results.multi_handedness):
                        continue
                        
                    hand_type = hands_results.multi_handedness[hand_idx].classification[0].label
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    
                    if hand_type == "Left":
                        left_hand = landmarks
                    else:
                        right_hand = landmarks
            
            all_frame_data["left_hand_landmarks"].append(left_hand)
            all_frame_data["right_hand_landmarks"].append(right_hand)
            
            # Record timestamp
            all_frame_data["frame_timestamps"].append(frame_time)
        
        cap.release()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save numerical data (.npy)
        output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(output_path, all_frame_data)
        
        # Create human-readable summary (.txt)
        txt_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.txt")
        
        with open(txt_output_path, "w") as f:
            f.write(f"Body Language Analysis for {video_file}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Video FPS: {fps}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n\n")
            
            # Summary statistics for posture
            f.write("POSTURE SUMMARY\n")
            f.write("-"*50 + "\n")
            
            valid_posture_frames = [p for p in all_frame_data["posture"] if p is not None]
            if valid_posture_frames:
                avg_torso_angle = np.mean([p["torso_angle"] for p in valid_posture_frames])
                avg_shoulder_sym = np.mean([p["shoulder_symmetry"] for p in valid_posture_frames])
                avg_hip_sym = np.mean([p["hip_symmetry"] for p in valid_posture_frames])
                
                f.write(f"Average torso angle: {avg_torso_angle:.2f} degrees\n")
                f.write(f"Average shoulder symmetry: {avg_shoulder_sym:.2f} degrees\n")
                f.write(f"Average hip symmetry: {avg_hip_sym:.2f} degrees\n")
                f.write(f"Average posture ratio: {np.mean([p['posture_ratio'] for p in valid_posture_frames]):.2f}\n\n")
            else:
                f.write("No valid posture data detected\n\n")
            
            # Summary statistics for facial cues
            f.write("FACIAL EXPRESSION SUMMARY\n")
            f.write("-"*50 + "\n")
            
            valid_face_frames = [fc for fc in all_frame_data["facial_cues"] if fc is not None]
            if valid_face_frames:
                avg_eye_open = np.mean([
                    (fc["left_eye_openness"] + fc["right_eye_openness"])/2 
                    for fc in valid_face_frames
                ])
                
                avg_eyebrow_raise = np.mean([
                    (fc["left_eyebrow_raise"] + fc["right_eyebrow_raise"])/2 
                    for fc in valid_face_frames
                ])
                
                avg_mouth_open = np.mean([fc["mouth_openness"] for fc in valid_face_frames])
                avg_mouth_width = np.mean([fc["mouth_width"] for fc in valid_face_frames])
                avg_smile = np.mean([fc["smile_indicator"] for fc in valid_face_frames])
                
                f.write(f"Average eye openness: {avg_eye_open:.4f}\n")
                f.write(f"Average eyebrow raise: {avg_eyebrow_raise:.4f}\n")
                f.write(f"Average mouth openness: {avg_mouth_open:.4f}\n")
                f.write(f"Average mouth width: {avg_mouth_width:.4f}\n")
                f.write(f"Average smile indicator: {avg_smile:.4f}\n\n")
            else:
                f.write("No valid facial data detected\n\n")
            
            # Summary statistics for movement
            f.write("MOVEMENT SUMMARY\n")
            f.write("-"*50 + "\n")
            
            valid_velocities = [v for v in all_frame_data["velocities"] if v is not None]
            if valid_velocities and len(valid_velocities) > 0:
                avg_velocity = np.mean([np.mean(v) for v in valid_velocities])
                max_velocity = np.max([np.max(v) if len(v) > 0 else 0 for v in valid_velocities])
                
                f.write(f"Average movement velocity: {avg_velocity:.4f}\n")
                f.write(f"Maximum movement velocity: {max_velocity:.4f}\n\n")
                
                # Identify high movement frames
                high_movement_frames = []
                for i, v in enumerate(valid_velocities):
                    if len(v) > 0 and np.mean(v) > avg_velocity * 1.5:
                        high_movement_frames.append(i)
                
                f.write(f"High movement detected in {len(high_movement_frames)} frames\n")
            else:
                f.write("No valid movement data detected\n")
            
            f.write("\nAnalysis complete.\n")

print("Processing complete!")
'''