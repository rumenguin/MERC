#!/usr/bin/env python3
"""
Audio Feature Extraction using OpenSmile for EmoReact Dataset

This script extracts comprehensive audio features from .wav files using OpenSmile's eGeMAPSv02 configuration.
Features are extracted with a frame length of 10ms and saved as comma-separated text files.
"""

import os
import subprocess
import sys
import time
from tqdm import tqdm  # For progress bar


def ensure_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def extract_features(input_file, output_file, opensmile_path, config_path, frame_length_ms=10):
    """
    Extract audio features using OpenSmile
    
    Args:
        input_file: Path to input .wav file
        output_file: Path to output .txt file
        opensmile_path: Path to OpenSmile binary
        config_path: Path to OpenSmile config file
        frame_length_ms: Frame length in milliseconds
    """
    # Convert frame_length_ms to seconds for OpenSmile
    frame_length_s = frame_length_ms / 1000.0
    
    # Create command for OpenSmile execution
    cmd = [
        opensmile_path,
        "-C", config_path,
        "-I", input_file,
        "-O", output_file,
        "-frameMode", "fixed",
        "-frameSize", str(frame_length_s),
        "-frameStep", str(frame_length_s),
        "-csvSeparator", ",",
        "-headerCSV", "1",
        "-appendCSV", "0",  # Overwrite existing files
        "-instname", os.path.basename(input_file).replace('.wav', ''),  # Instance name
    ]
    
    # Execute OpenSmile command
    try:
        # Hide stdout and stderr from OpenSmile to keep console clean
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError extracting features from {input_file}: {e}")
        print(f"Error output: {e.stderr.decode('utf-8')}")
        return False


def count_files(directory, extension='.wav'):
    """Count files with specific extension in directory"""
    count = 0
    for root, _, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(extension))
    return count


def process_dataset(dataset_path, output_path, opensmile_path, config_path):
    """
    Process all audio files in the dataset
    
    Args:
        dataset_path: Path to dataset containing Train, Test, Validation splits
        output_path: Path to save extracted features
        opensmile_path: Path to OpenSmile binary
        config_path: Path to OpenSmile config file
    """
    # Define splits
    splits = ["Train", "Test", "Validation"]
    output_splits = ["Train_feat", "Test_feat", "Val_feat"]
    
    # Count total files for progress tracking
    total_files = sum(count_files(os.path.join(dataset_path, split)) for split in splits)
    processed_files = 0
    successful_files = 0
    failed_files = 0
    
    print(f"Starting feature extraction for {total_files} audio files...")
    start_time = time.time()
    
    # Process each split
    for split, output_split in zip(splits, output_splits):
        input_dir = os.path.join(dataset_path, split)
        output_dir = os.path.join(output_path, output_split)
        
        # Create output directory if it doesn't exist
        ensure_directory(output_dir)
        
        # Get all .wav files in the input directory
        try:
            wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        except FileNotFoundError:
            print(f"Error: Directory not found: {input_dir}")
            continue
            
        print(f"\nProcessing {split} split: {len(wav_files)} files")
        
        # Process files with progress bar
        for wav_file in tqdm(wav_files, desc=f"Processing {split}"):
            input_file = os.path.join(input_dir, wav_file)
            output_file = os.path.join(output_dir, wav_file.replace('.wav', '.txt'))
            
            # Extract features
            success = extract_features(input_file, output_file, opensmile_path, config_path)
            processed_files += 1
            if success:
                successful_files += 1
            else:
                failed_files += 1
    
    # Calculate and display statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExtraction complete!")
    print(f"Total files processed: {processed_files}/{total_files}")
    print(f"Success: {successful_files}, Failed: {failed_files}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


def check_paths(opensmile_path, config_path, dataset_path):
    """Check if the required paths exist"""
    errors = []
    
    if not os.path.exists(opensmile_path):
        errors.append(f"OpenSmile binary not found at: {opensmile_path}")
    
    if not os.path.exists(config_path):
        errors.append(f"Configuration file not found at: {config_path}")
    
    if not os.path.exists(dataset_path):
        errors.append(f"Dataset directory not found at: {dataset_path}")
    
    return errors


def main():
    """Main function to run the audio feature extraction"""
    # Define paths
    dataset_path = "/Users/rumenguin/Research/MERC/EmoReact/Audio"
    output_path = "/Users/rumenguin/Research/MERC/EmoReact/AF"
    opensmile_path = "/Users/rumenguin/opensmile/build/progsrc/smilextract/SMILExtract"
    
    # eGeMAPSv02 configuration for emotion-related audio features
    config_path = "/Users/rumenguin/opensmile/config/egemaps/v02/eGeMAPSv02.conf"
    
    # Check if paths exist
    path_errors = check_paths(opensmile_path, config_path, dataset_path)
    if path_errors:
        for error in path_errors:
            print(f"Error: {error}")
        print("Please correct these errors and try again.")
        sys.exit(1)
    
    print("Audio Feature Extraction for EmoReact Dataset")
    print("--------------------------------------------")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Using configuration: {os.path.basename(config_path)}")
    print("--------------------------------------------")
    
    # Process the dataset
    process_dataset(dataset_path, output_path, opensmile_path, config_path)


if __name__ == "__main__":
    main()