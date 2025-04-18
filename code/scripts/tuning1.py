import os
import csv
import glob

# Base directories
transcript_base_dir = os.path.expanduser("~/Research/MERC/EmoReact/Transcript")
facial_base_dir = os.path.expanduser("~/Research/MERC/EmoReact/Facial")
output_file = os.path.expanduser("~/Research/MERC/EmoReact/facial_tuning.csv")

# Create a list to store all data
data = []

# Process each subdirectory (Train, Test, Validation)
for subdir in ["Train", "Test", "Validation"]:
    transcript_dir = os.path.join(transcript_base_dir, subdir)
    facial_dir = os.path.join(facial_base_dir, subdir)

    # Get all transcript files
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))

    for transcript_file in transcript_files:
        # Get the base filename
        base_filename = os.path.basename(transcript_file)
        video_name = os.path.splitext(base_filename)[0] + ".mp4"

        # Find the matching facial expression file
        facial_file = os.path.join(facial_dir, base_filename)

        # Check if both files exist
        if os.path.exists(facial_file):
            # Read the transcript content
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_content = f.read().strip()

            # Read the facial expression content
            with open(facial_file, "r", encoding="utf-8") as f:
                facial_content = f.read().strip()

            # Add to our data collection
            data.append([video_name, transcript_content, facial_content])

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Video", "Transcript", "Facial_Expression"])
    writer.writerows(data)

print(f"CSV file created successfully at {output_file}")
print(f"Total records processed: {len(data)}")
