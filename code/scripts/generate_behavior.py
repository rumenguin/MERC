import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set your paths
visual_features_root = "/Users/rumenguin/Research/MERC/EmoReact/Visual_features"
behavior_output_root = "/Users/rumenguin/Research/MERC/EmoReact/Behavior"
labels_root = "/Users/rumenguin/Research/MERC/EmoReact/Labels"

# Make sure Behavior folders exist
splits = ["Train_feat", "Test_feat", "Val_feat"]
behavior_splits = ["Train", "Test", "Validation"]

for split in behavior_splits:
    os.makedirs(os.path.join(behavior_output_root, split), exist_ok=True)


# Function to read and parse the custom txt file format
def read_custom_txt(file_path):
    try:
        # Read the file
        with open(file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return pd.DataFrame()

        # Parse the header (first line)
        headers = [h.strip() for h in lines[0].split(",")]

        # Parse data lines
        data = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                row_data = [val.strip() for val in line.split(",")]
                data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)

        # Convert numeric columns to float
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # Keep as string if conversion fails

        return df

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


# Function to load emotion labels and video names
def load_emotion_labels():
    # Dictionary to store all labels
    all_labels = {}

    # Load labels for each split
    for split_name in ["train", "test", "val"]:
        try:
            # Load video names
            names_file = os.path.join(
                labels_root, f"{split_name.capitalize()}_names.txt"
            )
            with open(names_file, "r") as f:
                video_names = [
                    name.strip().replace("'", "").replace(".mp4", "")
                    for name in f.readlines()
                ]

            # Load emotion labels
            labels_file = os.path.join(labels_root, f"{split_name}_labels.text")
            with open(labels_file, "r") as f:
                emotion_labels = [line.strip().split(",") for line in f.readlines()]

            # Convert to numerical values
            emotion_labels = [[float(val) for val in label] for label in emotion_labels]

            # Map video names to their emotion labels
            for i, name in enumerate(video_names):
                if i < len(emotion_labels):
                    all_labels[name] = emotion_labels[i]
        except Exception as e:
            print(f"Error loading {split_name} labels: {e}")

    return all_labels


# Function to get emotion names from binary labels
def get_emotions_from_labels(label):
    emotion_names = [
        "Curiosity",
        "Uncertainty",
        "Excitement",
        "Happiness",
        "Surprise",
        "Disgust",
        "Fear",
        "Frustration",
    ]

    # Get emotions with value 1
    active_emotions = [emotion_names[i] for i in range(8) if label[i] == 1]

    # Get valence score (last value)
    valence = label[8]

    return active_emotions, valence


# Enhanced gaze direction function
def analyze_gaze(gaze_df):
    # Average all gaze vectors
    avg_x = (gaze_df["x_0"].mean() + gaze_df["x_1"].mean()) / 2
    avg_y = (gaze_df["y_0"].mean() + gaze_df["y_1"].mean()) / 2
    avg_z = (gaze_df["z_0"].mean() + gaze_df["z_1"].mean()) / 2

    # Also analyze head direction (if available)
    has_head_data = all(
        col in gaze_df.columns
        for col in ["x_h0", "y_h0", "z_h0", "x_h1", "y_h1", "z_h1"]
    )
    if has_head_data:
        head_x = (gaze_df["x_h0"].mean() + gaze_df["x_h1"].mean()) / 2
        head_y = (gaze_df["y_h0"].mean() + gaze_df["y_h1"].mean()) / 2

    # Calculate gaze variability over time
    if len(gaze_df) > 5:  # Only if we have enough frames
        x_variability = (
            gaze_df["x_0"].diff().abs().mean() + gaze_df["x_1"].diff().abs().mean()
        )
        y_variability = (
            gaze_df["y_0"].diff().abs().mean() + gaze_df["y_1"].diff().abs().mean()
        )

        if x_variability > 0.05 or y_variability > 0.05:
            gaze_consistency = "shifting"
        else:
            gaze_consistency = "steady"
    else:
        # Check for gaze consistency using standard deviation as fallback
        gaze_x_std = np.std([gaze_df["x_0"].mean(), gaze_df["x_1"].mean()])
        gaze_y_std = np.std([gaze_df["y_0"].mean(), gaze_df["y_1"].mean()])
        gaze_consistency = (
            "shifting" if (gaze_x_std > 0.1 or gaze_y_std > 0.1) else "steady"
        )

    # Basic direction
    if avg_x < -0.2:
        horizontal = "left"
    elif avg_x > 0.2:
        horizontal = "right"
    else:
        horizontal = "center"

    if avg_y < -0.2:
        vertical = "up"
    elif avg_y > 0.2:
        vertical = "down"
    else:
        vertical = "center"

    # Depth perception (based on z values)
    focus = (
        "focused on something distant"
        if avg_z < -0.95
        else "focused on something nearby"
    )

    # Generate comprehensive gaze description
    if horizontal == "center" and vertical == "center":
        basic_direction = "looking straight ahead"
    elif vertical == "center":
        basic_direction = f"looking to the {horizontal}"
    elif horizontal == "center":
        basic_direction = f"looking {vertical}"
    else:
        basic_direction = f"looking {vertical} and to the {horizontal}"

    # Combine all gaze information
    gaze_description = f"{basic_direction} with a {gaze_consistency} gaze, {focus}"

    # Add head-eye coordination info if available
    if has_head_data:
        if abs(head_x - avg_x) < 0.1 and abs(head_y - avg_y) < 0.1:
            gaze_description += ", with head and eyes aligned"
        else:
            gaze_description += ", with head and eyes oriented differently"

    return gaze_description


# Function to analyze facial AUs in detail with emotion focus
def analyze_facial_aus(au_df, emotions=None):
    au_means = au_df.select_dtypes(include=["number"]).mean()
    expressions = []

    # Map all AUs to potential expressions
    # Upper face AUs
    if "AU01_r" in au_means.index and au_means["AU01_r"] > 0.7:
        if "Surprise" in emotions:
            expressions.append("raising inner eyebrows (indicating surprise)")
        elif "Fear" in emotions:
            expressions.append("raising inner eyebrows (indicating fear)")
        elif "Curiosity" in emotions:
            expressions.append("raising inner eyebrows (indicating curiosity)")
        else:
            expressions.append("raising inner eyebrows")

    if "AU02_r" in au_means.index and au_means["AU02_r"] > 0.7:
        if "Surprise" in emotions:
            expressions.append("raising outer eyebrows (indicating surprise)")
        elif "Fear" in emotions:
            expressions.append("raising outer eyebrows (indicating fear)")
        else:
            expressions.append("raising outer eyebrows")

    if "AU04_r" in au_means.index and au_means["AU04_r"] > 0.7:
        if "Frustration" in emotions:
            expressions.append("furrowing brows (indicating frustration)")
        elif "Uncertainty" in emotions:
            expressions.append("furrowing brows (indicating uncertainty)")
        else:
            expressions.append("furrowing brows")

    if "AU05_r" in au_means.index and au_means["AU05_r"] > 0.7:
        if "Surprise" in emotions:
            expressions.append("raising upper eyelids (indicating surprise)")
        elif "Fear" in emotions:
            expressions.append("raising upper eyelids (indicating fear)")
        else:
            expressions.append("raising upper eyelids")

    if "AU06_r" in au_means.index and au_means["AU06_r"] > 0.7:
        if "Happiness" in emotions:
            expressions.append(
                "raising cheeks in a genuine smile (indicating happiness)"
            )
        elif "Excitement" in emotions:
            expressions.append("raising cheeks (indicating excitement)")
        else:
            expressions.append("raising cheeks")

    if "AU07_r" in au_means.index and au_means["AU07_r"] > 0.7:
        if "Uncertainty" in emotions:
            expressions.append("squinting eyes (indicating uncertainty)")
        elif "Curiosity" in emotions:
            expressions.append("narrowing eyes (indicating focused curiosity)")
        else:
            expressions.append("squinting eyes")

    # Lower face AUs
    if "AU09_r" in au_means.index and au_means["AU09_r"] > 0.7:
        if "Disgust" in emotions:
            expressions.append("wrinkling nose (indicating disgust)")
        else:
            expressions.append("wrinkling nose")

    if "AU10_r" in au_means.index and au_means["AU10_r"] > 0.7:
        if "Disgust" in emotions:
            expressions.append("raising upper lip (indicating disgust)")
        else:
            expressions.append("raising upper lip")

    if "AU12_r" in au_means.index and au_means["AU12_r"] > 0.7:
        if "Happiness" in emotions:
            expressions.append("smiling (indicating happiness)")
        elif "Excitement" in emotions:
            expressions.append("smiling (indicating excitement)")
        else:
            expressions.append("smiling")

    if "AU14_r" in au_means.index and au_means["AU14_r"] > 0.7:
        if "Uncertainty" in emotions:
            expressions.append("dimpling (indicating skepticism or uncertainty)")
        else:
            expressions.append("dimpling")

    if "AU15_r" in au_means.index and au_means["AU15_r"] > 0.7:
        if "Frustration" in emotions:
            expressions.append("pulling lip corners down (indicating frustration)")
        elif "Fear" in emotions:
            expressions.append("pulling lip corners down (indicating fear)")
        else:
            expressions.append("pulling lip corners down")

    if "AU17_r" in au_means.index and au_means["AU17_r"] > 0.7:
        if "Uncertainty" in emotions:
            expressions.append("raising chin (indicating doubt or uncertainty)")
        else:
            expressions.append("raising chin")

    if "AU20_r" in au_means.index and au_means["AU20_r"] > 0.7:
        if "Fear" in emotions:
            expressions.append("stretching lips horizontally (indicating fear)")
        elif "Surprise" in emotions:
            expressions.append("stretching lips horizontally (indicating surprise)")
        else:
            expressions.append("stretching lips horizontally")

    if "AU23_c" in au_means.index and au_means["AU23_c"] > 0.5:
        if "Frustration" in emotions:
            expressions.append("tightening lips (indicating restraint or frustration)")
        else:
            expressions.append("tightening lips")

    if "AU25_r" in au_means.index and au_means["AU25_r"] > 0.7:
        if "Surprise" in emotions:
            expressions.append("parting lips (indicating surprise)")
        else:
            expressions.append("parting lips")

    if "AU26_r" in au_means.index and au_means["AU26_r"] > 0.7:
        if "Surprise" in emotions:
            expressions.append("dropping jaw (indicating surprise)")
        else:
            expressions.append("dropping jaw")

    if "AU28_c" in au_means.index and au_means["AU28_c"] > 0.5:
        if "Uncertainty" in emotions:
            expressions.append("sucking lips inward (indicating uncertainty)")
        else:
            expressions.append("sucking lips inward")

    if "AU45_c" in au_means.index and au_means["AU45_c"] > 0.3:
        expressions.append("blinking frequently")

    return expressions


# Function to analyze head pose parameters with emotional context
def analyze_head_pose(pose_df, emotions=None):
    # Get mean values of all pose parameters
    avg_rx = pose_df["rx"].mean()
    avg_ry = pose_df["ry"].mean()
    avg_rz = pose_df["rz"].mean()

    # Calculate variation in head movement
    rx_std = pose_df["rx"].std()
    ry_std = pose_df["ry"].std()
    rz_std = pose_df["rz"].std()

    # Determine head movements with emotional context
    movements = []

    # Static head orientation with emotional interpretations based on labeled emotions
    if avg_rx > 0.2:
        movement = "tilting their head to the right"
        if emotions and ("Curiosity" in emotions):
            movement += " (consistent with their curiosity)"
        movements.append(movement)
    elif avg_rx < -0.2:
        movement = "tilting their head to the left"
        if emotions and ("Curiosity" in emotions):
            movement += " (consistent with their curiosity)"
        movements.append(movement)

    if avg_ry > 0.2:
        movement = "nodding downwards"
        if emotions and ("Uncertainty" in emotions):
            movement += " (possibly reflecting their uncertainty)"
        movements.append(movement)
    elif avg_ry < -0.2:
        movement = "raising their head upwards"
        if emotions and ("Excitement" in emotions or "Surprise" in emotions):
            movement += f" (consistent with their {emotions[0] if len(emotions) > 0 else 'emotional state'})"
        movements.append(movement)

    if avg_rz > 0.2:
        movement = "turning their head to the right"
        if emotions and ("Curiosity" in emotions):
            movement += " (possibly exploring their environment with curiosity)"
        movements.append(movement)
    elif avg_rz < -0.2:
        movement = "turning their head to the left"
        if emotions and ("Curiosity" in emotions):
            movement += " (possibly exploring their environment with curiosity)"
        movements.append(movement)

    # Head movement patterns with emotional context
    head_stability = ""
    if rx_std > 0.1 or ry_std > 0.1 or rz_std > 0.1:
        if rx_std > ry_std and rx_std > rz_std:
            head_stability = "frequently tilting their head side to side"
            if emotions and ("Uncertainty" in emotions):
                head_stability += " (reflecting their uncertainty)"
        elif ry_std > rx_std and ry_std > rz_std:
            head_stability = "frequently nodding up and down"
            if emotions and any(e in emotions for e in ["Excitement", "Happiness"]):
                head_stability += " (expressing their enthusiasm)"
        elif rz_std > rx_std and rz_std > ry_std:
            head_stability = "frequently turning their head left to right"
            if emotions and "Curiosity" in emotions:
                head_stability += " (showing their curiosity)"
        else:
            head_stability = "showing frequent head movements"
            if emotions and any(e in emotions for e in ["Excitement", "Surprise"]):
                head_stability += f" (consistent with their {emotions[0] if len(emotions) > 0 else 'emotion'})"
    else:
        head_stability = "maintaining a stable head position"
        if emotions and any(e in emotions for e in ["Fear", "Surprise"]):
            head_stability += " (possibly frozen in their emotional state)"

    if head_stability:
        movements.append(head_stability)

    return movements


# Enhanced behavior generation with labeled emotions
def generate_behavior(params_path, au_path, gaze_path, emotion_labels, video_name):
    try:
        # Use custom function to read the txt files
        df_pose = read_custom_txt(params_path)
        df_au = read_custom_txt(au_path)
        df_gaze = read_custom_txt(gaze_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return "No behavior detected."

    if df_pose.empty or df_au.empty or df_gaze.empty:
        return "No behavior detected."

    # Filter high confidence frames, only if confidence column exists
    if "confidence" in df_pose.columns:
        df_pose = df_pose[df_pose["confidence"] > 0.9]
    if "confidence" in df_au.columns:
        df_au = df_au[df_au["confidence"] > 0.9]
    if "confidence" in df_gaze.columns:
        df_gaze = df_gaze[df_gaze["confidence"] > 0.9]

    if len(df_pose) == 0 or len(df_au) == 0 or len(df_gaze) == 0:
        return "No behavior detected."

    # Get emotion labels for this video
    emotions = []
    valence = 0

    # If we have labels for this video, use them
    base_name_for_lookup = (
        video_name.replace("_au.txt", "")
        .replace("_gaze.txt", "")
        .replace(".params.txt", "")
    )
    if base_name_for_lookup in emotion_labels:
        emotions, valence = get_emotions_from_labels(
            emotion_labels[base_name_for_lookup]
        )

    # Get detailed gaze direction
    gaze_description = analyze_gaze(df_gaze)

    # Get detailed facial expressions with emotion context
    facial_expressions = analyze_facial_aus(df_au, emotions)

    # Get head pose analysis with emotion context
    head_movements = analyze_head_pose(df_pose, emotions)

    # Combine all behavioral observations
    behavior_text = ""

    # Start with emotions if available
    if emotions:
        if len(emotions) == 1:
            behavior_text += f"The person is expressing {emotions[0].lower()}. "
        elif len(emotions) > 1:
            last_emotion = emotions[-1]
            other_emotions = emotions[:-1]
            behavior_text += f"The person is showing a complex emotional state of {', '.join(e.lower() for e in other_emotions)} and {last_emotion.lower()}. "

    # Add valence if available
    if valence > 0:
        valence_description = ""
        if valence < 3:
            valence_description = "slightly negative"
        elif valence < 4:
            valence_description = "neutral"
        elif valence < 5:
            valence_description = "mildly positive"
        elif valence < 6:
            valence_description = "quite positive"
        else:
            valence_description = "very positive"

        behavior_text += f"Their overall emotional valence is {valence_description}. "

    # Add facial expressions
    if facial_expressions:
        behavior_text += "They are showing " + ", ".join(facial_expressions) + ". "

    # Add head movements
    if head_movements:
        behavior_text += "Their head position shows " + ", ".join(head_movements) + ". "

    # Add gaze description
    behavior_text += f"Their eyes are {gaze_description}. "

    # If no specific behaviors detected
    if not behavior_text:
        if emotions:
            primary_emotion = emotions[0].lower()
            behavior_text = f"The person is expressing {primary_emotion} with minimal facial cues or movements."
        else:
            behavior_text = "The person has a neutral expression and posture, showing minimal emotional cues."

    return behavior_text


# Main process
def main():
    # Load emotion labels first
    print("Loading emotion labels...")
    emotion_labels = load_emotion_labels()
    print(f"Loaded labels for {len(emotion_labels)} videos")

    # Process each split
    for vsplit, bsplit in zip(splits, behavior_splits):
        input_folder = os.path.join(visual_features_root, vsplit)
        output_folder = os.path.join(behavior_output_root, bsplit)

        # Check if directory exists before processing
        if not os.path.exists(input_folder):
            print(f"Warning: Input folder {input_folder} does not exist. Skipping.")
            continue

        print(f"Processing files in {input_folder}")
        files = [f for f in os.listdir(input_folder) if f.endswith(".params.txt")]

        for file in tqdm(files, desc=f"Processing {bsplit}"):
            base_name = file.replace(".params.txt", "")
            params_path = os.path.join(input_folder, file)
            au_path = os.path.join(input_folder, base_name + "_au.txt")
            gaze_path = os.path.join(input_folder, base_name + "_gaze.txt")

            # Check if all required files exist
            if not os.path.exists(au_path) or not os.path.exists(gaze_path):
                print(f"Warning: Missing files for {base_name}. Skipping.")
                continue

            try:
                behavior_description = generate_behavior(
                    params_path, au_path, gaze_path, emotion_labels, base_name
                )

                # Save behavior description
                filename = base_name + ".txt"
                output_path = os.path.join(output_folder, filename)

                with open(output_path, "w") as f:
                    f.write(behavior_description)
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
                # Log the error in more detail
                import traceback

                print(traceback.format_exc())


# Execute the main process
if __name__ == "__main__":
    main()
