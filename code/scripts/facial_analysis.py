import os
import re
import pandas as pd
import numpy as np
from glob import glob

# Paths
base_dir = os.path.expanduser("~/Research/MERC/EmoReact")
behaviors_dir = os.path.join(base_dir, "Behaviors")
labels_dir = os.path.join(base_dir, "Labels")
facial_dir = os.path.join(base_dir, "Facial")

# Create facial directories if they don't exist
for split in ["Train", "Test", "Validation"]:
    os.makedirs(os.path.join(facial_dir, split), exist_ok=True)


# Helper function to read labels
def read_labels(split):
    folder_split = "Val" if split == "Validation" else split
    label_file = os.path.join(labels_dir, f"{folder_split.lower()}_labels.text")
    labels_df = pd.read_csv(label_file, header=None)
    # The last column is valence (index 8)
    valence_series = labels_df.iloc[:, -1]

    # Extract filenames from behaviors directory
    behavior_files = glob(os.path.join(behaviors_dir, split, "*.txt"))
    file_names = [os.path.basename(f) for f in behavior_files]

    # Create dictionary with filename -> valence mapping
    return dict(zip(file_names, valence_series))


# Function to categorize valence score (1-7 scale)
def categorize_valence(valence):
    if valence < 2:
        return "very negative"
    elif valence < 3:
        return "slightly negative"
    elif valence < 4:
        return "neutral"
    elif valence < 5:
        return "mildly positive"
    elif valence < 6:
        return "quite positive"
    else:
        return "very positive"


# Extract behavior patterns from a string description
def extract_behavior_patterns(behavior_str):
    patterns = {
        "facial_expressions": [],
        "head_position": [],
        "gaze": [],
        "other_details": [],
    }

    # Facial expressions patterns
    facial_expr_patterns = [
        "frowning",
        "smiling",
        "showing genuine smile",
        "showing sadness",
        "raising inner eyebrows",
        "raising outer eyebrows",
        "widening eyes",
        "dimpling cheeks",
        "blinking",
        "wrinkling nose",
        "raising upper lip",
        "looking somewhat sad",
        "raising chin",
        "stretching lips",
        "parting lips",
        "dropping jaw",
        "sucking lips",
    ]

    for pattern in facial_expr_patterns:
        if pattern in behavior_str.lower():
            patterns["facial_expressions"].append(pattern)

    # Head position patterns
    head_pos_patterns = [
        "stable head position",
        "turning head left",
        "turning head right",
        "tilting head upwards",
        "tilting head right",
        "tilting head left",
        "nodding downwards",
        "neutral head pose",
    ]

    for pattern in head_pos_patterns:
        if pattern in behavior_str.lower():
            patterns["head_position"].append(pattern)

    # Gaze patterns
    gaze_patterns = [
        "looking to the right",
        "looking to the left",
        "looking upward",
        "looking downward",
        "with a focused gaze",
        "with a relaxed gaze",
        "looking straight ahead",
        "maintaining eye contact",
    ]

    for pattern in gaze_patterns:
        if pattern in behavior_str.lower():
            patterns["gaze"].append(pattern)

    # Other facial details
    detail_patterns = [
        "face appearing wider than average",
        "face appearing narrower than average",
        "distinctive mouth shape",
        "asymmetric facial expression",
        "puffed cheeks",
        "tensed middle face",
        "subtle facial expressions",
    ]

    for pattern in detail_patterns:
        if pattern in behavior_str.lower():
            patterns["other_details"].append(pattern)

    return patterns


# Function to analyze emotional valence trend across sections
def analyze_emotional_trend(sections_patterns):
    # Define emotion valence for different expressions
    expression_valence = {
        "frowning": -1,
        "showing sadness": -1,
        "looking somewhat sad": -0.5,
        "raising inner eyebrows": 0,  # Neutral/surprise
        "raising outer eyebrows": 0,  # Neutral/surprise
        "widening eyes": 0,  # Neutral/surprise
        "smiling": 1,
        "showing genuine smile": 1.5,
        "dimpling cheeks": 0.5,
        "blinking": 0,
        "wrinkling nose": -0.5,
        "dropping jaw": 0,
        "parting lips": 0,
    }

    section_valences = []

    for patterns in sections_patterns:
        section_score = 0
        expression_count = 0

        for expr in patterns["facial_expressions"]:
            if expr in expression_valence:
                section_score += expression_valence[expr]
                expression_count += 1

        # Normalize the score
        if expression_count > 0:
            section_score /= expression_count

        section_valences.append(section_score)

    # Determine emotional trend
    if len(section_valences) >= 3:
        if section_valences[0] < 0 and section_valences[-1] > 0:
            return "negative to positive"
        elif section_valences[0] > 0 and section_valences[-1] < 0:
            return "positive to negative"
        elif all(v < 0 for v in section_valences):
            return "consistently negative"
        elif all(v > 0 for v in section_valences):
            return "consistently positive"
        elif all(abs(v) < 0.2 for v in section_valences):
            return "consistently neutral"
        else:
            return "mixed emotions"

    return "unclear emotional progression"


# Function to summarize a section's behaviors
def summarize_section_behaviors(patterns):
    # Count frequency of each pattern type
    facial_expr_counts = {}
    for expr in patterns["facial_expressions"]:
        if expr in facial_expr_counts:
            facial_expr_counts[expr] += 1
        else:
            facial_expr_counts[expr] = 1

    head_pos_counts = {}
    for pos in patterns["head_position"]:
        if pos in head_pos_counts:
            head_pos_counts[pos] += 1
        else:
            head_pos_counts[pos] = 1

    gaze_counts = {}
    for gaze in patterns["gaze"]:
        if gaze in gaze_counts:
            gaze_counts[gaze] += 1
        else:
            gaze_counts[gaze] = 1

    detail_counts = {}
    for detail in patterns["other_details"]:
        if detail in detail_counts:
            detail_counts[detail] += 1
        else:
            detail_counts[detail] = 1

    # Get dominant patterns (top 2 for facial expressions, top 1 for others)
    dominant_facial = sorted(
        facial_expr_counts.items(), key=lambda x: x[1], reverse=True
    )[:2]
    dominant_facial = [item[0] for item in dominant_facial if item[1] > 0]

    dominant_head = sorted(head_pos_counts.items(), key=lambda x: x[1], reverse=True)[
        :1
    ]
    dominant_head = [item[0] for item in dominant_head if item[1] > 0]

    dominant_gaze = sorted(gaze_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    dominant_gaze = [item[0] for item in dominant_gaze if item[1] > 0]

    dominant_detail = sorted(detail_counts.items(), key=lambda x: x[1], reverse=True)[
        :1
    ]
    dominant_detail = [item[0] for item in dominant_detail if item[1] > 0]

    # Combine into a summary pattern
    summary = {
        "facial_expressions": dominant_facial,
        "head_position": dominant_head,
        "gaze": dominant_gaze,
        "other_details": dominant_detail,
    }

    return summary


# Function to create a natural language description from section patterns
def create_section_description(section_name, patterns):
    description = f"The {section_name}, the child "

    # Add facial expressions
    if patterns["facial_expressions"]:
        description += "is " + " and ".join(patterns["facial_expressions"])
    else:
        description += "maintains a neutral expression"

    # Add head position
    if patterns["head_position"]:
        description += " while " + patterns["head_position"][0]

    # Add gaze information
    if patterns["gaze"]:
        description += ", " + " and ".join(patterns["gaze"])

    # Add other details
    if patterns["other_details"]:
        description += ", with " + patterns["other_details"][0]

    description += "."

    return description


# Function to analyze facial expressions in a file
def analyze_facial_expressions(file_path, valence):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip header line
    data_lines = [line for line in lines if not line.startswith("frame")]

    # If no valid data, return empty analysis
    if not data_lines:
        return f"No valid facial expression data found.\n\nTheir overall emotional valence is {categorize_valence(valence)}."

    # Extract behavior descriptions
    behaviors = []
    for line in data_lines:
        parts = line.strip().split(", ")
        if len(parts) >= 5:  # Make sure we have enough parts
            behavior_text = parts[4]  # Behavior is the 5th element
            behaviors.append(behavior_text)

    # Divide into three parts
    total_frames = len(behaviors)
    first_third = behaviors[: total_frames // 3]
    middle_third = behaviors[total_frames // 3 : 2 * total_frames // 3]
    last_third = behaviors[2 * total_frames // 3 :]

    # Extract and analyze behavior patterns from each section
    first_patterns = [extract_behavior_patterns(behavior) for behavior in first_third]
    middle_patterns = [extract_behavior_patterns(behavior) for behavior in middle_third]
    last_patterns = [extract_behavior_patterns(behavior) for behavior in last_third]

    # Summarize each section
    first_summary = summarize_section_behaviors(merge_patterns(first_patterns))
    middle_summary = summarize_section_behaviors(merge_patterns(middle_patterns))
    last_summary = summarize_section_behaviors(merge_patterns(last_patterns))

    # Create natural language descriptions for each section
    first_description = create_section_description("video starts with", first_summary)
    middle_description = create_section_description(
        "middle portion shows", middle_summary
    )
    last_description = create_section_description(
        "final third of the video shows", last_summary
    )

    # Analyze emotional trend
    emotional_trend = analyze_emotional_trend(
        [first_summary, middle_summary, last_summary]
    )

    # Generate complete summary
    # summary = "Facial Expression Summary:\n"
    summary = first_description + " " + middle_description + " " + last_description

    # Add emotional progression
    summary += (
        f"\n\nThe emotional journey progresses with {emotional_trend} expressions."
    )

    # Add valence sentence
    valence_category = categorize_valence(valence)
    summary += f"\n\nTheir overall emotional valence is {valence_category}."

    return summary


# Helper function to merge multiple frame patterns
def merge_patterns(patterns_list):
    merged = {
        "facial_expressions": [],
        "head_position": [],
        "gaze": [],
        "other_details": [],
    }

    # Combine all patterns
    for patterns in patterns_list:
        merged["facial_expressions"].extend(patterns["facial_expressions"])
        merged["head_position"].extend(patterns["head_position"])
        merged["gaze"].extend(patterns["gaze"])
        merged["other_details"].extend(patterns["other_details"])

    return merged


# Process each split
def process_all_splits():
    for split in ["Train", "Test", "Validation"]:
        print(f"Processing {split} split...")

        # Get valence labels
        valence_dict = read_labels(split)

        # Process each file in the behaviors directory
        behavior_files = glob(os.path.join(behaviors_dir, split, "*.txt"))

        for behavior_file in behavior_files:
            file_name = os.path.basename(behavior_file)

            # Get valence for this file
            valence = valence_dict.get(
                file_name, 3.5
            )  # Default to neutral if not found

            # Analyze facial expressions
            summary = analyze_facial_expressions(behavior_file, valence)

            # Write to output file
            output_file = os.path.join(facial_dir, split, file_name)
            with open(output_file, "w") as f:
                f.write(summary)

            print(f"Processed {file_name}")


if __name__ == "__main__":
    # Process all splits
    process_all_splits()
    print("Facial expression analysis and summarization completed!")
