import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')


# Define paths
base_input_dir = "/Users/rumenguin/Research/MERC/EmoReact/Audio"
base_output_dir = "/Users/rumenguin/Research/MERC/EmoReact/Audio_features"

# Create output directories if they don't exist
for split in ["Train", "Test", "Validation"]:
    output_subdir = split.replace("Validation", "Val") + "_feat"
    os.makedirs(os.path.join(base_output_dir, output_subdir), exist_ok=True)

# Function to extract audio features
def extract_audio_features(audio_file_path, frame_length_ms=16):
    """
    Extract comprehensive audio features with specified frame length
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file
    frame_length_ms : int
        Frame length in milliseconds
    
    Returns:
    --------
    features_dict : dict
        Dictionary containing all extracted features
    """
    # Load audio file
    y, sr = librosa.load(audio_file_path, sr=None)  # Use original sampling rate
    
    # Convert frame length from ms to samples
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = frame_length  # Non-overlapping frames
    
    # Calculate duration in seconds
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract features
    features_dict = {}
    
    # Basic features
    features_dict['sr'] = sr
    features_dict['duration'] = duration
    features_dict['frame_length_ms'] = frame_length_ms
    
    # Time-domain features
    features_dict['rms'] = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    features_dict['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Spectral features
    stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    
    features_dict['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    features_dict['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    features_dict['spectral_contrast'] = librosa.feature.spectral_contrast(S=stft, sr=sr).T
    features_dict['spectral_flatness'] = librosa.feature.spectral_flatness(S=stft).T
    features_dict['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    features_dict['mfcc'] = mfccs.T
    
    # Delta and Delta-Delta (velocity and acceleration) of MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    features_dict['mfcc_delta'] = mfcc_delta.T
    features_dict['mfcc_delta2'] = mfcc_delta2.T
    
    # Chroma features
    features_dict['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length).T
    features_dict['chroma_cqt'] = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T
    
    # Mel-scaled spectrogram
    mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=max(512, frame_length), hop_length=hop_length, n_mels=40, fmax=sr // 2)

    features_dict['mel_spectrogram'] = mel_spec.T
    features_dict['log_mel_spectrogram'] = librosa.power_to_db(mel_spec).T
    
    # Pitch (F0) and harmonic features
    f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz('E3'), fmax=librosa.note_to_hz('C6'),
    frame_length=max(512, frame_length), hop_length=hop_length)
    features_dict['f0'] = f0
    features_dict['voiced_flag'] = voiced_flag
    features_dict['voiced_probability'] = voiced_probs
    
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features_dict['harmonic_rms'] = librosa.feature.rms(y=y_harmonic, frame_length=frame_length, hop_length=hop_length)[0]
    features_dict['percussive_rms'] = librosa.feature.rms(y=y_percussive, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Tempo and beat information
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    features_dict['tempo'] = tempo
    features_dict['beats'] = beats
    
    # Calculate frame timestamps in seconds
    num_frames = len(features_dict['rms'])
    features_dict['frame_times'] = np.arange(num_frames) * (frame_length_ms / 1000)
    
    return features_dict

# Process all files in the dataset
splits = ["Train", "Test", "Validation"]

for split in splits:
    input_dir = os.path.join(base_input_dir, split)
    output_subdir = split.replace("Validation", "Val") + "_feat"
    output_dir = os.path.join(base_output_dir, output_subdir)
    
    # Get all wav files in the directory
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Processing {len(audio_files)} files in {split} set...")
    
    for audio_file in tqdm(audio_files):
        input_path = os.path.join(input_dir, audio_file)
        output_base = os.path.join(output_dir, os.path.splitext(audio_file)[0])
        
        # Extract features
        try:
            features = extract_audio_features(input_path, frame_length_ms=16)
            
            # Save features in .npy format
            # Create a structured numpy array with frame-level features
            frame_features = np.column_stack([
                features['frame_times'],
                features['rms'],
                features['zero_crossing_rate'],
                features['spectral_centroid'],
                features['spectral_bandwidth'],
                features['spectral_rolloff'],
                features['f0'],
                features['voiced_probability'],
                features['harmonic_rms'],
                features['percussive_rms'],
                features['mfcc'],
                features['mfcc_delta'],
                features['chroma_stft']
            ])
            np.save(f"{output_base}.npy", frame_features)
            
            # Save feature information in .txt format (replacing the JSON summary)
            with open(f"{output_base}.txt", 'w') as f:
                f.write(f"Filename: {audio_file}\n")
                f.write(f"Sampling rate: {features['sr']}\n")
                f.write(f"Duration: {features['duration']}\n")
                f.write(f"Number of frames: {len(features['frame_times'])}\n")
                f.write(f"Frame length (ms): {features['frame_length_ms']}\n")
                f.write(f"Mean RMS: {float(np.mean(features['rms']))}\n")
                f.write(f"Std RMS: {float(np.std(features['rms']))}\n")
                
                if np.all(np.isnan(features['f0'])):
                    f.write("Mean F0: NaN\n")
                    f.write("Std F0: NaN\n")
                else:
                    f.write(f"Mean F0: {float(np.nanmean(features['f0']))}\n")
                    f.write(f"Std F0: {float(np.nanstd(features['f0']))}\n")
                
                f.write(f"Std F0: {float(np.nanstd(features['f0']))}\n")
                f.write(f"Tempo: {float(features['tempo'])}\n")
                
                f.write("Mean MFCCs:\n")
                for i in range(features['mfcc'].shape[1]):
                    f.write(f"  MFCC_{i+1}: {float(np.mean(features['mfcc'][:, i]))}\n")
                
                f.write("Mean Chroma:\n")
                for i in range(features['chroma_stft'].shape[1]):
                    f.write(f"  Chroma_{i+1}: {float(np.mean(features['chroma_stft'][:, i]))}\n")
                
            print(f"Successfully processed {audio_file}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")

print("Feature extraction completed!")