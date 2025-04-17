import os
import csv
import pandas as pd

# Define base directory
base_dir = "/Users/rumenguin/Research/MERC/EmoReact"

# Define emotion labels
emotions = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

# Function to read video names from file
def read_video_names(file_path):
    with open(file_path, 'r') as f:
        # Remove the single quotes surrounding the video names
        return [line.strip().strip("'") for line in f]

# Function to read labels from file
def read_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) >= 9:  # Make sure we have 9 values (8 emotions + valence)
                emotion_values = [int(float(val)) for val in values[:8]]
                valence = float(values[8])
                labels.append((emotion_values, valence))
    return labels

# Function to read text from file
def read_text_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

# Process each split (Train, Test, Val)
dataset = []
id_counter = 0
for split in ["Train", "Test", "Val"]:
    # Read video names
    names_file = os.path.join(base_dir, "Labels", f"{split}_names.txt")
    video_names = read_video_names(names_file)
    
    # Read labels
    labels_file = os.path.join(base_dir, "Labels", f"{split.lower()}_labels.text")
    all_labels = read_labels(labels_file)
    
    # Use full "Validation" for folder paths when split is "Val"
    folder_split = "Validation" if split == "Val" else split
    
    # Process each video
    for i, video_name_with_ext in enumerate(video_names):
        if i >= len(all_labels):
            print(f"Warning: No labels found for {video_name_with_ext}")
            continue
            
        # Extract video name without extension for finding transcript/behavior files
        video_name = video_name_with_ext.replace('.mp4', '')
        
        # Get emotion labels and valence
        emotion_values, valence = all_labels[i]
        
        # Convert binary values to emotion names
        present_emotions = [emotions[j] for j in range(8) if emotion_values[j] == 1]
        
        # If no emotions are present, use "None"
        emotion_str = ", ".join(present_emotions) if present_emotions else "None"
        
        # Get transcript and behavior
        transcript_file = os.path.join(base_dir, "Transcript", folder_split, f"{video_name}.txt")
        behavior_file = os.path.join(base_dir, "Behavior", folder_split, f"{video_name}.txt")
        
        transcript = read_text_file(transcript_file)
        behavior = read_text_file(behavior_file)
        
        # Add to dataset
        dataset.append({
            "ID": id_counter,
            "Video": video_name_with_ext,
            "Transcript": transcript,
            "Behavior": behavior,
            "Labels": emotion_str,
            "Valence": valence
        })
        id_counter += 1

# Create DataFrame and save to CSV
df = pd.DataFrame(dataset)
csv_path = os.path.join(base_dir, "dataset.csv")
df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Dataset created successfully with {len(dataset)} entries.")
print(f"Saved to: {csv_path}")