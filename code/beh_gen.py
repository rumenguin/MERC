import os
import glob
import pandas as pd


def load_file(filepath):
    """Load data file and return as pandas DataFrame."""
    try:
        df = pd.read_csv(filepath, sep=",", skipinitialspace=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_natural_behavior_description(
    facial_expression, head_position, gaze_description, shape_details
):
    """
    Create a natural language description of behavior from the detected components.

    Args:
        facial_expression: List of facial expression descriptions
        head_position: List of head position descriptions
        gaze_description: List of gaze descriptions
        shape_details: List of shape parameter details

    Returns:
        A natural language description that sounds like a good sentence
    """
    # Get the most important facial expressions (limit to 2-3 for natural flow)
    primary_expressions = []
    secondary_expressions = []

    # Prioritize significant expressions
    priority_expressions = [
        "frowning",
        "showing genuine smile",
        "smiling",
        "showing sadness",
        "raising inner eyebrows",
        "widening eyes",
    ]

    # Sort expressions by priority
    for expr in facial_expression:
        if any(priority in expr for priority in priority_expressions):
            primary_expressions.append(expr)
        else:
            secondary_expressions.append(expr)

    # Limit to most important expressions
    if len(primary_expressions) > 2:
        facial_expressions_to_use = primary_expressions[:2]
    elif primary_expressions:
        facial_expressions_to_use = primary_expressions
        if secondary_expressions and len(facial_expressions_to_use) < 2:
            facial_expressions_to_use.append(secondary_expressions[0])
    elif secondary_expressions:
        facial_expressions_to_use = secondary_expressions[:2]
    else:
        facial_expressions_to_use = ["maintaining a neutral expression"]

    # Get key head position (limit to 1 for natural flow)
    key_head_position = (
        head_position[0] if head_position else "with a neutral head pose"
    )

    # Get key gaze (limit to the most meaningful 1-2)
    key_gaze = []
    if gaze_description:
        directional_gaze = [
            g
            for g in gaze_description
            if any(
                dir in g
                for dir in [
                    "looking to the right",
                    "looking to the left",
                    "looking upward",
                    "looking downward",
                ]
            )
        ]
        if directional_gaze:
            key_gaze.append(directional_gaze[0])

        intensity_gaze = [
            g
            for g in gaze_description
            if any(int in g for int in ["focused gaze", "relaxed gaze"])
        ]
        if intensity_gaze:
            key_gaze.append(intensity_gaze[0])

    if not key_gaze:
        key_gaze = ["maintaining eye contact"]

    # Get most relevant shape details (limit to 1 for natural flow)
    key_shape = shape_details[0] if shape_details else ""

    # Compose the behavior description using natural language patterns
    description_parts = []

    # Start with subject (Person/The child) performing action
    subject = "The child is"

    # Build the main action phrase with primary expressions
    if facial_expressions_to_use:
        main_action = " " + " and ".join(facial_expressions_to_use)
        description_parts.append(subject + main_action)
    else:
        description_parts.append(subject + " displaying a neutral expression")

    # Add head position with transition
    description_parts.append(" while " + key_head_position)

    # Add gaze information with appropriate connector
    if key_gaze:
        gaze_phrase = ", " + " and ".join(key_gaze)
        description_parts.append(gaze_phrase)

    # Add shape detail if meaningful
    if key_shape:
        description_parts.append(f", with {key_shape}")

    # Combine all parts into a coherent sentence
    behavior = "".join(description_parts) + "."

    # Capitalize first letter
    behavior = behavior[0].upper() + behavior[1:]

    # Ensure there are no double spaces or awkward punctuation
    behavior = behavior.replace("  ", " ").replace(" ,", ",")

    return behavior


def detect_behavior(au_data, gaze_data, params_data, row_idx):
    """
    Detect behavior based on combined features from AU, gaze, and params data.
    Generates natural language descriptions of facial behaviors.
    Incorporates p0-p33 shape parameters for detailed facial expression analysis.
    """
    # Get row data from each file
    if (
        row_idx >= len(au_data)
        or row_idx >= len(gaze_data)
        or row_idx >= len(params_data)
    ):
        return "Subject appears outside of frame or cannot be analyzed."

    au_row = au_data.iloc[row_idx]
    gaze_row = gaze_data.iloc[row_idx]
    params_row = params_data.iloc[row_idx]

    # Initialize behavior description components
    facial_expression = []
    head_position = []
    gaze_description = []
    shape_details = []

    # ---- FACIAL EXPRESSION ANALYSIS (AUs) ----
    # Inner brow raiser (AU01)
    if "AU01_r" in au_row and au_row["AU01_r"] > 0.7:
        facial_expression.append("raising inner eyebrows")

    # Outer brow raiser (AU02)
    if "AU02_r" in au_row and au_row["AU02_r"] > 0.7:
        facial_expression.append("raising outer eyebrows")

    # Brow lowerer (AU04) - frowning
    if "AU04_r" in au_row and au_row["AU04_r"] > 0.7:
        facial_expression.append("frowning")
    elif "AU04_c" in au_row and au_row["AU04_c"] > 0:
        facial_expression.append("slightly frowning")

    # Upper lid raiser (AU05) - eyes widening
    if "AU05_r" in au_row and au_row["AU05_r"] > 0.7:
        facial_expression.append("widening eyes")

    # Cheek raiser (AU06) - genuine smile
    if "AU06_r" in au_row and au_row["AU06_r"] > 0.7:
        facial_expression.append("showing genuine smile")

    # Nose wrinkler (AU09)
    if "AU09_r" in au_row and au_row["AU09_r"] > 0.7:
        facial_expression.append("wrinkling nose")

    # Upper lip raiser (AU10)
    if "AU10_r" in au_row and au_row["AU10_r"] > 0.7:
        facial_expression.append("raising upper lip")

    # Lip corner puller (AU12) - smiling
    if "AU12_r" in au_row and au_row["AU12_r"] > 0.7:
        facial_expression.append("smiling")
    elif "AU12_c" in au_row and au_row["AU12_c"] > 0:
        facial_expression.append("slightly smiling")

    # Dimpler (AU14)
    if "AU14_r" in au_row and au_row["AU14_r"] > 0.7:
        facial_expression.append("dimpling cheeks")

    # Lip corner depressor (AU15) - sad expression
    if "AU15_r" in au_row and au_row["AU15_r"] > 0.7:
        facial_expression.append("showing sadness")
    elif "AU15_c" in au_row and au_row["AU15_c"] > 0:
        facial_expression.append("looking somewhat sad")

    # Chin raiser (AU17)
    if "AU17_r" in au_row and au_row["AU17_r"] > 0.7:
        facial_expression.append("raising chin")

    # Lip stretcher (AU20)
    if "AU20_r" in au_row and au_row["AU20_r"] > 0.7:
        facial_expression.append("stretching lips horizontally")

    # Lips part (AU25)
    if "AU25_r" in au_row and au_row["AU25_r"] > 0.7:
        facial_expression.append("parting lips")

    # Jaw drop (AU26)
    if "AU26_r" in au_row and au_row["AU26_r"] > 0.7:
        facial_expression.append("dropping jaw")

    # Lip suck (AU28)
    if "AU28_c" in au_row and au_row["AU28_c"] > 0:
        facial_expression.append("sucking lips")

    # Blink (AU45)
    if "AU45_c" in au_row and au_row["AU45_c"] > 0:
        facial_expression.append("blinking")

    # ---- HEAD POSITION ANALYSIS ----
    # Head rotation analysis
    rx = params_row.get("rx", 0)
    ry = params_row.get("ry", 0)
    rz = params_row.get("rz", 0)

    # Vertical head position (nodding)
    if rx > 0.1:
        head_position.append("nodding downwards")
    elif rx < -0.1:
        head_position.append("tilting head upwards")

    # Horizontal head position (shaking)
    if ry > 0.1:
        head_position.append("turning head right")
    elif ry < -0.1:
        head_position.append("turning head left")

    # Head tilt (sideways)
    if rz > 0.1:
        head_position.append("tilting head right")
    elif rz < -0.1:
        head_position.append("tilting head left")

    # If head is relatively still
    if abs(rx) < 0.05 and abs(ry) < 0.05 and abs(rz) < 0.05:
        head_position.append("maintaining a stable head position")

    # ---- GAZE ANALYSIS ----
    # Eye gaze direction
    x_0 = gaze_row.get("x_0", 0)  # Right eye x direction
    y_0 = gaze_row.get("y_0", 0)  # Right eye y direction
    x_1 = gaze_row.get("x_1", 0)  # Left eye x direction
    y_1 = gaze_row.get("y_1", 0)  # Left eye y direction

    # Head-based gaze features
    x_h0 = gaze_row.get("x_h0", 0)  # Right eye head-based x direction
    y_h0 = gaze_row.get("y_h0", 0)  # Right eye head-based y direction
    z_h0 = gaze_row.get("z_h0", 0)  # Right eye head-based z direction
    x_h1 = gaze_row.get("x_h1", 0)  # Left eye head-based x direction
    y_h1 = gaze_row.get("y_h1", 0)  # Left eye head-based y direction
    z_h1 = gaze_row.get("z_h1", 0)  # Left eye head-based z direction

    # Horizontal gaze - combine eye and head-based features
    avg_x = (x_0 + x_1) / 2
    avg_x_h = (x_h0 + x_h1) / 2
    # Weight the combined gaze direction (70% eye-based, 30% head-based)
    combined_x = 0.7 * avg_x + 0.3 * avg_x_h

    if combined_x > 0.2:
        gaze_description.append("looking to the right")
    elif combined_x < -0.2:
        gaze_description.append("looking to the left")
    else:
        gaze_description.append("looking straight ahead horizontally")

    # Vertical gaze - combine eye and head-based features
    avg_y = (y_0 + y_1) / 2
    avg_y_h = (y_h0 + y_h1) / 2
    # Weight the combined gaze direction (70% eye-based, 30% head-based)
    combined_y = 0.7 * avg_y + 0.3 * avg_y_h

    if combined_y > 0.2:
        gaze_description.append("looking upward")
    elif combined_y < -0.2:
        gaze_description.append("looking downward")
    else:
        gaze_description.append("at eye level")

    # Gaze intensity/focus - combine eye and head-based z components
    gaze_z = (gaze_row.get("z_0", -1) + gaze_row.get("z_1", -1)) / 2
    gaze_z_h = (z_h0 + z_h1) / 2  # Head-based depth component
    # Weight the combined gaze depth (60% eye-based, 40% head-based)
    combined_z = 0.6 * gaze_z + 0.4 * gaze_z_h

    if combined_z < -0.95:
        gaze_description.append("with a focused gaze")
    else:
        gaze_description.append("with a relaxed gaze")

    # Check for gaze-head coordination
    head_dir = "center"
    if ry > 0.1:
        head_dir = "right"
    elif ry < -0.1:
        head_dir = "left"

    gaze_dir = "center"
    if combined_x > 0.2:
        gaze_dir = "right"
    elif combined_x < -0.2:
        gaze_dir = "left"

    # Check if there's a significant mismatch between eye and head-based gaze
    eye_head_x_diff = abs(avg_x - avg_x_h)
    if eye_head_x_diff > 0.3:
        gaze_description.append("with eyes and head orientation misaligned")

    # Check if head and overall gaze directions match
    if head_dir != gaze_dir and head_dir != "center" and gaze_dir != "center":
        gaze_description.append("with head and eyes oriented differently")

    # ---- SHAPE PARAMETERS ANALYSIS (p0-p33) ----
    # Extract shape parameters (p0 through p33)
    shape_params = {}
    for i in range(34):
        param_name = f"p{i}"
        if param_name in params_row:
            shape_params[param_name] = params_row[param_name]

    # Extract head pose parameters
    head_pose_params = {
        "rx": params_row.get("rx", 0),  # Head rotation X (pitch)
        "ry": params_row.get("ry", 0),  # Head rotation Y (yaw)
        "rz": params_row.get("rz", 0),  # Head rotation Z (roll)
        "tx": params_row.get("tx", 0),  # Head translation X
        "ty": params_row.get("ty", 0),  # Head translation Y
        "scale": params_row.get("scale", 1),  # Head scale factor
    }

    # Group 1: p0-p5 (jaw width, face height)
    if "p0" in shape_params and abs(shape_params["p0"]) > 10:
        if shape_params["p0"] > 0:
            shape_details.append("face appearing wider than average")
        else:
            shape_details.append("face appearing narrower than average")

    # Group 2: p6-p10 (mouth shape and position)
    mouth_deformation = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(6, 11))
    if mouth_deformation > 5:
        shape_details.append("distinctive mouth shape")

        # Detect specific mouth shapes
        if "p6" in shape_params and shape_params["p6"] > 3:
            shape_details.append("lips pursed forward")
        elif "p6" in shape_params and shape_params["p6"] < -3:
            shape_details.append("lips retracted")

        if "p7" in shape_params and shape_params["p7"] > 2:
            shape_details.append("mouth widened")
        elif "p7" in shape_params and shape_params["p7"] < -2:
            shape_details.append("mouth narrowed")

    # Group 3: p11-p15 (eyes and brows)
    eyes_deformation = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(11, 16))
    if eyes_deformation > 5:
        # Detect specific eye movements not captured by AUs
        if "p11" in shape_params and shape_params["p11"] > 2:
            shape_details.append("eyes more open than usual")
        elif "p11" in shape_params and shape_params["p11"] < -2:
            shape_details.append("eyes more closed than usual")

        if "p12" in shape_params and abs(shape_params["p12"]) > 2:
            shape_details.append("asymmetric eye expression")

    # Group 4: p16-p20 (nose region)
    nose_deformation = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(16, 21))
    if nose_deformation > 4:
        if "p16" in shape_params and abs(shape_params["p16"]) > 2:
            shape_details.append("noticeable nose movement")

    # Group 5: p21-p25 (cheeks and middle face)
    cheek_deformation = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(21, 26))
    if cheek_deformation > 3:
        if "p21" in shape_params and shape_params["p21"] > 1:
            shape_details.append("puffed cheeks")
        elif "p22" in shape_params and shape_params["p22"] > 1:
            shape_details.append("tensed middle face")

    # Group 6: p26-p33 (subtle expressions)
    subtle_deformation = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(26, 34))
    if subtle_deformation > 2:
        shape_details.append("subtle facial expressions")

        # Check for asymmetry
        asymmetry_score = sum(abs(shape_params.get(f"p{i}", 0)) for i in range(30, 34))
        if asymmetry_score > 1:
            shape_details.append("asymmetric facial expression")

    # Head pose analysis using previously extracted parameters
    head_pose_details = []

    # Analyze head translation (tx, ty)
    tx = head_pose_params["tx"]
    ty = head_pose_params["ty"]

    # Horizontal displacement
    if abs(tx) > 10:
        if tx > 0:
            head_pose_details.append("head shifted right")
        else:
            head_pose_details.append("head shifted left")

    # Vertical displacement
    if abs(ty) > 10:
        if ty > 0:
            head_pose_details.append("head positioned higher")
        else:
            head_pose_details.append("head positioned lower")

    # Analyze head scale
    scale = head_pose_params["scale"]
    if scale > 1.2:
        head_pose_details.append("head closer to camera")
    elif scale < 0.8:
        head_pose_details.append("head further from camera")

    # Analyze combined head pose patterns
    # Check for nodding-like movement pattern (combine rx with ty)
    if abs(head_pose_params["rx"]) > 0.15 and abs(ty) > 5:
        if (head_pose_params["rx"] > 0 and ty < 0) or (
            head_pose_params["rx"] < 0 and ty > 0
        ):
            head_pose_details.append("nodding movement")

    # Check for shaking-like movement pattern (combine ry with tx)
    if abs(head_pose_params["ry"]) > 0.15 and abs(tx) > 5:
        if (head_pose_params["ry"] > 0 and tx > 0) or (
            head_pose_params["ry"] < 0 and tx < 0
        ):
            head_pose_details.append("head shaking movement")

    # Add head pose details to shape details
    if head_pose_details:
        shape_details.extend(head_pose_details)

    # Create a natural language description of the behavior
    behavior = create_natural_behavior_description(
        facial_expression, head_position, gaze_description, shape_details
    )

    return behavior


def process_video_files(au_path, gaze_path, params_path, output_path):
    """Process feature files for a single video and output behavior file."""
    # Load data files
    au_data = load_file(au_path)
    gaze_data = load_file(gaze_path)
    params_data = load_file(params_path)

    if au_data is None or gaze_data is None or params_data is None:
        print(
            f"Skipping video due to loading errors: {os.path.basename(au_path).replace('_au.txt', '')}"
        )
        return

    # Initialize behavior data
    behavior_data = []

    # Process each frame
    for i in range(len(au_data)):
        if i >= len(gaze_data) or i >= len(params_data):
            break

        frame = au_data.iloc[i]["frame"]
        timestamp = au_data.iloc[i]["timestamp"]
        confidence = au_data.iloc[i]["confidence"]
        success = au_data.iloc[i]["success"]

        # Skip if frame numbers don't match
        if frame != gaze_data.iloc[i]["frame"] or frame != params_data.iloc[i]["frame"]:
            print(f"Frame mismatch at index {i} for video {os.path.basename(au_path)}")
            continue

        behavior = detect_behavior(au_data, gaze_data, params_data, i)

        behavior_data.append([frame, timestamp, confidence, success, behavior])

    # Create DataFrame for manipulation
    columns = ["frame", "timestamp", "confidence", "success", "behavior"]
    behavior_df = pd.DataFrame(behavior_data, columns=columns)

    # Write to text file with comma separation
    with open(output_path, "w") as f:
        # Write header
        f.write("frame, timestamp, confidence, success, behavior\n")

        # Write data rows
        for _, row in behavior_df.iterrows():
            f.write(
                f"{int(row['frame'])}, {row['timestamp']}, {row['confidence']}, {int(row['success'])}, {row['behavior']}\n"
            )

    print(f"Created behavior file: {output_path}")


def process_all_videos(input_dir, output_dir):
    """Process all videos in the input directory structure."""
    # Create output directory structure
    splits = ["Train", "Test", "Validation"]
    input_splits = ["Train_feat", "Test_feat", "Val_feat"]

    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Process each split
    for i, split in enumerate(input_splits):
        split_dir = os.path.join(input_dir, split)
        output_split = splits[i if i < 2 else 2]  # Map Val_feat to Validation

        # Get all AU files (each represents a video)
        au_files = glob.glob(os.path.join(split_dir, "*_au.txt"))

        for au_path in au_files:
            # Determine paths for all files
            video_name = os.path.basename(au_path).replace("_au.txt", "")
            gaze_path = os.path.join(split_dir, f"{video_name}_gaze.txt")
            params_path = os.path.join(split_dir, f"{video_name}.params.txt")

            # Check if all files exist
            if not os.path.exists(gaze_path) or not os.path.exists(params_path):
                print(f"Missing files for video {video_name}")
                continue

            # Define output path
            output_path = os.path.join(output_dir, output_split, f"{video_name}.txt")

            # Process video files
            process_video_files(au_path, gaze_path, params_path, output_path)


def main():
    # Define paths
    input_dir = "/Users/rumenguin/Research/MERC/EmoReact/Visual_features"
    output_dir = "/Users/rumenguin/Research/MERC/EmoReact/Behaviors"

    # Process all videos
    process_all_videos(input_dir, output_dir)
    print("Behavior extraction completed!")


if __name__ == "__main__":
    main()

