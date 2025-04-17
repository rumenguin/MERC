import os
from moviepy.editor import VideoFileClip
import whisper
from tqdm import tqdm

# Paths
root_dir = "/Users/rumenguin/Research/MERC/EmoReact"
splits = ["Train", "Test", "Validation"]

# Load Whisper model once
model = whisper.load_model("small")  # You can use "medium" if you want even better quality

# Create Audio/ and Text/ folders
for split in splits:
    os.makedirs(os.path.join(root_dir, "Audio", split), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "Text", split), exist_ok=True)

# Process each split
for split in splits:
    print(f"Processing {split} videos...")
    data_split_path = os.path.join(root_dir, "Data", split)
    audio_split_path = os.path.join(root_dir, "Audio", split)
    text_split_path = os.path.join(root_dir, "Text", split)

    video_files = [f for f in os.listdir(data_split_path) if f.endswith(".mp4")]

    for video_file in tqdm(video_files, desc=f"{split}"):
        video_path = os.path.join(data_split_path, video_file)
        
        # Prepare output paths
        filename_wo_ext = os.path.splitext(video_file)[0]
        audio_output_path = os.path.join(audio_split_path, filename_wo_ext + ".wav")
        text_output_path = os.path.join(text_split_path, filename_wo_ext + ".txt")
        
        try:
            # 1. Extract audio
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_output_path, fps=16000, codec='pcm_s16le', verbose=False, logger=None)

            # 2. Transcribe audio to text
            result = model.transcribe(audio_output_path)
            transcript = result["text"]

            # 3. Save transcript
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(transcript.strip())

        except Exception as e:
            print(f"Failed for {video_file}: {e}")
