import os
import whisper
from tqdm import tqdm

# Paths
root_dir = "/Users/rumenguin/Research/MERC/EmoReact"
splits = ["Train", "Test", "Validation"]

# Load Whisper model (large for best quality)
print("Loading Whisper large model...")
model = whisper.load_model("large")

# Create Transcript/ folders
for split in splits:
    os.makedirs(os.path.join(root_dir, "Transcript", split), exist_ok=True)

# Process each split
for split in splits:
    print(f"Processing {split} audio files...")
    audio_split_path = os.path.join(root_dir, "Audio", split)
    transcript_split_path = os.path.join(root_dir, "Transcript", split)

    audio_files = [f for f in os.listdir(audio_split_path) if f.endswith(".wav")]

    for audio_file in tqdm(audio_files, desc=f"{split}"):
        audio_path = os.path.join(audio_split_path, audio_file)
        
        # Prepare output path
        filename_wo_ext = os.path.splitext(audio_file)[0]
        transcript_output_path = os.path.join(transcript_split_path, filename_wo_ext + ".txt")
        
        try:
            # Transcribe audio to text
            result = model.transcribe(audio_path)
            transcript = result["text"]

            # Save transcript
            with open(transcript_output_path, "w", encoding="utf-8") as f:
                f.write(transcript.strip())

        except Exception as e:
            print(f"Failed to transcribe {audio_file}: {e}")
