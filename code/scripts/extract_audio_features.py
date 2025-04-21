"""
import parselmouth
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    filename=r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features/processing_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories
base_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio"
output_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features"
folders = ["Train", "Test", "Validation"]
output_folders = ["Train_feat", "Test_feat", "Val_feat"]

# Parameters
sr = 16000  # Sample rate
time_step = 0.01  # 10 ms frame length
min_duration = 0.05  # Min 50 ms
energy_threshold = 1e-8  # Relaxed for EmoReact

# Voice Activity Detection (VAD)
def apply_vad(audio, sr):
    frame_len = int(0.02 * sr)  # 20 ms frames
    frame_shift = int(0.01 * sr)  # 10 ms shift
    energy = []
    for i in range(0, len(audio) - frame_len + 1, frame_shift):
        frame = audio[i:i + frame_len]
        energy.append(np.mean(frame ** 2))

    thresh = np.mean(energy) * 0.1
    speech_frames = [e > thresh for e in energy]

    speech_audio = np.zeros_like(audio)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * frame_shift
            end = min(start + frame_len, len(audio))
            speech_audio[start:end] = audio[start:end]

    # Trim zeros
    non_zero = np.where(speech_audio != 0)[0]
    if len(non_zero) > 0:
        speech_audio = speech_audio[non_zero[0]:non_zero[-1] + 1]
    if len(speech_audio) < sr * min_duration:
        return audio  # Fallback to original
    return speech_audio

# Helper function for jitter and shimmer extraction
def extract_jitter_shimmer(sound, file_name):
    try:
        # Create Pitch object optimized for voice analysis
        pitch = sound.to_pitch(0.75/75, 75, 600)

        # Create PointProcess object from Sound and Pitch
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        # Check if we have enough points for analysis
        num_points = parselmouth.praat.call(point_process, "Get number of points")

        if num_points >= 3:
            # Calculate jitter using local method
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

            # Calculate shimmer - this needs both Sound and PointProcess objects
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # Additional check for unreasonable values
            if jitter > 1.0:  # Jitter values are typically very small (<0.1)
                logging.warning(f"File {file_name}: Very high jitter value ({jitter}), capping at 1.0")
                jitter = 1.0

            if shimmer > 1.0:  # Shimmer values typically <1.0
                logging.warning(f"File {file_name}: Very high shimmer value ({shimmer}), capping at 1.0")
                shimmer = 1.0

            logging.info(f"File {file_name}: Successfully extracted jitter ({jitter:.4f}) and shimmer ({shimmer:.4f})")
            return jitter, shimmer
        else:
            logging.warning(f"File {file_name}: Insufficient pulses for jitter/shimmer ({num_points} points found)")
            return 0.0, 0.0
    except Exception as e:
        logging.error(f"Failed to compute jitter/shimmer for {file_name}: {e}")
        return 0.0, 0.0

# Process audio files
for i, folder in enumerate(folders):
    input_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, output_folders[i])
    os.makedirs(output_folder, exist_ok=True)

    for file in Path(input_folder).glob("*.wav"):
        try:
            print(f"Processing {file.name}")
            logging.info(f"Processing {file.name}")

            # Read audio
            audio, sr_file = sf.read(file)
            duration = len(audio) / sr_file

            # Validate duration
            if duration < min_duration:
                print(f"File {file.name} too short ({duration:.2f} s), skipping")
                logging.error(f"File {file.name} too short ({duration:.2f} s), skipped")
                continue

            # Limit duration (for very long files)
            if duration > 30:  # Cap at 30 seconds max
                audio = audio[:int(30 * sr_file)]
                logging.info(f"File {file.name} truncated to 30 seconds")

            # Convert to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Resample to 16 kHz
            if sr_file != sr:
                audio = resample(audio, int(len(audio) * sr / sr_file))

            # Normalize
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9
            else:
                print(f"File {file.name} has zero amplitude, skipping")
                logging.error(f"File {file.name} has zero amplitude, skipped")
                continue

            # Apply VAD
            audio = apply_vad(audio, sr)
            if len(audio) < sr * min_duration:
                print(f"No significant speech in {file.name} after VAD, skipping")
                logging.error(f"No significant speech in {file.name} after VAD, skipped")
                continue

            # Compute energy
            energy = np.mean(audio ** 2)
            logging.info(f"File {file.name}: Duration {duration:.2f} s, Energy {energy:.6e}")
            if energy < energy_threshold:
                print(f"File {file.name} has low energy ({energy:.6e}), proceeding")
                logging.warning(f"File {file.name} has low energy ({energy:.6e}), proceeding")

            # Create Praat sound object
            snd = parselmouth.Sound(audio, sampling_frequency=sr)

            # Extract features
            pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
            f0 = pitch.selected_array['frequency']

            intensity = snd.to_intensity(minimum_pitch=75, time_step=time_step)
            intensity_vals = intensity.values[0]

            hnr = snd.to_harmonicity(time_step=time_step, minimum_pitch=75)
            hnr_vals = hnr.values[0]

            formants = snd.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=5500)
            time_points = formants.xs()
            f1 = np.zeros(len(time_points))
            f2 = np.zeros(len(time_points))
            for t_idx, t in enumerate(time_points):
                try:
                    f1[t_idx] = formants.get_value_at_time(1, t, unit="HERTZ")
                    f2[t_idx] = formants.get_value_at_time(2, t, unit="HERTZ")
                except:
                    f1[t_idx] = np.nan
                    f2[t_idx] = np.nan
            # Replace NaNs with interpolated values or zeros
            f1 = pd.Series(f1).interpolate().fillna(0).values
            f2 = pd.Series(f2).interpolate().fillna(0).values

            # Extract jitter and shimmer using our helper function
            jitter, shimmer = extract_jitter_shimmer(snd, file.name)

            # Align lengths
            min_len = min(len(f0), len(intensity_vals), len(hnr_vals), len(f1), len(f2))
            features = np.vstack([
                f0[:min_len],
                intensity_vals[:min_len],
                hnr_vals[:min_len],
                f1[:min_len],
                f2[:min_len],
                np.full(min_len, jitter),
                np.full(min_len, shimmer)
            ]).T

            # Validate features
            if np.all(features[:, :5] == 0):  # Check F0, intensity, HNR, F1, F2
                print(f"Features for {file.name} are all zeros, skipping")
                logging.error(f"Features for {file.name} are all zeros, skipped")
                continue

            # Save to CSV
            output_file = os.path.join(output_folder, file.stem + ".csv")
            pd.DataFrame(
                features,
                columns=["f0", "intensity", "hnr", "formant1", "formant2", "jitter", "shimmer"]
            ).to_csv(output_file, index=False)

            print(f"Successfully processed {file.name}")
            logging.info(f"Successfully processed {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            logging.error(f"Error processing {file.name}: {e}")
            continue

print("Feature extraction complete! See C:\MERC\Audio_features\processing_log.txt for details.")
logging.info("Feature extraction complete!")
"""


'''
import parselmouth
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import os
from pathlib import Path
import logging
from scipy import fft
from scipy.signal import stft

# Setup logging
logging.basicConfig(
    filename=r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features/processing_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories
base_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio"
output_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features"
folders = ["Train", "Test", "Validation"]
output_folders = ["Train_feat", "Test_feat", "Val_feat"]

# Parameters
sr = 16000  # Sample rate
time_step = 0.01  # 10 ms frame length
min_duration = 0.05  # Min 50 ms
energy_threshold = 1e-8  # Relaxed for EmoReact

# Voice Activity Detection (VAD)
def apply_vad(audio, sr):
    frame_len = int(0.02 * sr)  # 20 ms frames
    frame_shift = int(0.01 * sr)  # 10 ms shift
    energy = []
    for i in range(0, len(audio) - frame_len + 1, frame_shift):
        frame = audio[i:i + frame_len]
        energy.append(np.mean(frame ** 2))
    
    thresh = np.mean(energy) * 0.1
    speech_frames = [e > thresh for e in energy]
    
    speech_audio = np.zeros_like(audio)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * frame_shift
            end = min(start + frame_len, len(audio))
            speech_audio[start:end] = audio[start:end]
    
    # Trim zeros
    non_zero = np.where(speech_audio != 0)[0]
    if len(non_zero) > 0:
        speech_audio = speech_audio[non_zero[0]:non_zero[-1] + 1]
    if len(speech_audio) < sr * min_duration:
        return audio  # Fallback to original
    return speech_audio

# Helper functions for additional features
def extract_spectral_centroid(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract spectral centroid."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    centroids = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        if np.sum(frame**2) < 1e-10:  # Skip silent frames
            centroids.append(0)
            continue
            
        # Calculate FFT
        spectrum = np.abs(fft.rfft(frame * np.hamming(len(frame))))
        freqs = fft.rfftfreq(frame_size, 1/sr)
        
        # Calculate centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            centroids.append(centroid)
        else:
            centroids.append(0)
    
    return np.array(centroids)

def extract_zero_crossing_rate(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract zero crossing rate."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    zcr_values = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame).astype(int))))
        zcr = zero_crossings / (frame_size - 1)
        zcr_values.append(zcr)
    
    return np.array(zcr_values)

def extract_spectral_flux(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract spectral flux - measures the change in magnitude in the frequency spectrum."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    # Compute STFT
    frequencies, times, Zxx = stft(audio, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    
    # Calculate magnitude spectrum
    mag_spec = np.abs(Zxx)
    
    # Calculate flux (difference between consecutive spectra)
    flux = np.zeros(mag_spec.shape[1])
    for i in range(1, mag_spec.shape[1]):
        # Euclidean distance between consecutive spectra
        flux[i] = np.sqrt(np.sum((mag_spec[:, i] - mag_spec[:, i-1])**2))
    
    # Normalize
    if np.max(flux) > 0:
        flux = flux / np.max(flux)
    
    return flux

def extract_spectral_rolloff(audio, sr, frame_len=0.02, hop_len=0.01, rolloff_percent=0.85):
    """Extract spectral rolloff - frequency below which rolloff_percent of spectrum energy lies."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    rolloffs = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        if np.sum(frame**2) < 1e-10:  # Skip silent frames
            rolloffs.append(0)
            continue
            
        # Calculate FFT
        spectrum = np.abs(fft.rfft(frame * np.hamming(len(frame))))
        freqs = fft.rfftfreq(frame_size, 1/sr)
        
        # Calculate rolloff
        if np.sum(spectrum) > 0:
            cumulative_sum = np.cumsum(spectrum)
            threshold = rolloff_percent * cumulative_sum[-1]
            # Find the frequency that corresponds to rolloff_percent of energy
            for j, energy in enumerate(cumulative_sum):
                if energy >= threshold:
                    rolloffs.append(freqs[j])
                    break
            else:
                rolloffs.append(0)
        else:
            rolloffs.append(0)
    
    return np.array(rolloffs)

# Helper function for jitter and shimmer extraction
def extract_jitter_shimmer(sound, file_name):
    try:
        # Create Pitch object optimized for voice analysis
        pitch = sound.to_pitch(0.75/75, 75, 600)
        
        # Create PointProcess object from Sound and Pitch
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        
        # Check if we have enough points for analysis
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        
        if num_points >= 3:
            # Calculate jitter using local method
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Calculate shimmer - this needs both Sound and PointProcess objects
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Additional check for unreasonable values
            if jitter > 1.0:  # Jitter values are typically very small (<0.1)
                logging.warning(f"File {file_name}: Very high jitter value ({jitter}), capping at 1.0")
                jitter = 1.0
                
            if shimmer > 1.0:  # Shimmer values typically <1.0
                logging.warning(f"File {file_name}: Very high shimmer value ({shimmer}), capping at 1.0")
                shimmer = 1.0
                
            logging.info(f"File {file_name}: Successfully extracted jitter ({jitter:.4f}) and shimmer ({shimmer:.4f})")
            return jitter, shimmer
        else:
            logging.warning(f"File {file_name}: Insufficient pulses for jitter/shimmer ({num_points} points found)")
            return 0.0, 0.0
    except Exception as e:
        logging.error(f"Failed to compute jitter/shimmer for {file_name}: {e}")
        return 0.0, 0.0

# Process audio files
for i, folder in enumerate(folders):
    input_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, output_folders[i])
    os.makedirs(output_folder, exist_ok=True)
    
    for file in Path(input_folder).glob("*.wav"):
        try:
            print(f"Processing {file.name}")
            logging.info(f"Processing {file.name}")
            
            # Read audio
            audio, sr_file = sf.read(file)
            duration = len(audio) / sr_file
            
            # Validate duration
            if duration < min_duration:
                print(f"File {file.name} too short ({duration:.2f} s), skipping")
                logging.error(f"File {file.name} too short ({duration:.2f} s), skipped")
                continue
            
            # Limit duration (for very long files)
            if duration > 30:  # Cap at 30 seconds max
                audio = audio[:int(30 * sr_file)]
                logging.info(f"File {file.name} truncated to 30 seconds")
            
            # Convert to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16 kHz
            if sr_file != sr:
                audio = resample(audio, int(len(audio) * sr / sr_file))
            
            # Normalize
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9
            else:
                print(f"File {file.name} has zero amplitude, skipping")
                logging.error(f"File {file.name} has zero amplitude, skipped")
                continue
            
            # Apply VAD
            audio = apply_vad(audio, sr)
            if len(audio) < sr * min_duration:
                print(f"No significant speech in {file.name} after VAD, skipping")
                logging.error(f"No significant speech in {file.name} after VAD, skipped")
                continue
            
            # Compute energy
            energy = np.mean(audio ** 2)
            logging.info(f"File {file.name}: Duration {duration:.2f} s, Energy {energy:.6e}")
            if energy < energy_threshold:
                print(f"File {file.name} has low energy ({energy:.6e}), proceeding")
                logging.warning(f"File {file.name} has low energy ({energy:.6e}), proceeding")
            
            # Create Praat sound object
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Extract base features
            pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
            f0 = pitch.selected_array['frequency']
            
            intensity = snd.to_intensity(minimum_pitch=75, time_step=time_step)
            intensity_vals = intensity.values[0]
            
            hnr = snd.to_harmonicity(time_step=time_step, minimum_pitch=75)
            hnr_vals = hnr.values[0]
            
            formants = snd.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=5500)
            time_points = formants.xs()
            f1 = np.zeros(len(time_points))
            f2 = np.zeros(len(time_points))
            for t_idx, t in enumerate(time_points):
                try:
                    f1[t_idx] = formants.get_value_at_time(1, t, unit="HERTZ")
                    f2[t_idx] = formants.get_value_at_time(2, t, unit="HERTZ")
                except:
                    f1[t_idx] = np.nan
                    f2[t_idx] = np.nan
            # Replace NaNs with interpolated values or zeros
            f1 = pd.Series(f1).interpolate().fillna(0).values
            f2 = pd.Series(f2).interpolate().fillna(0).values
            
            # Extract jitter and shimmer using our helper function
            jitter, shimmer = extract_jitter_shimmer(snd, file.name)
            
            # Safe extraction of additional features with individual try-except blocks
            try:
                spec_centroid = extract_spectral_centroid(audio, sr)
                print(f"  ✓ Spectral centroid: {len(spec_centroid)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral centroid for {file.name}: {e}")
                spec_centroid = np.zeros_like(f0)
            
            try:
                zcr = extract_zero_crossing_rate(audio, sr)
                print(f"  ✓ Zero crossing rate: {len(zcr)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract zero crossing rate for {file.name}: {e}")
                zcr = np.zeros_like(f0)
            
            try:
                spec_flux = extract_spectral_flux(audio, sr)
                print(f"  ✓ Spectral flux: {len(spec_flux)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral flux for {file.name}: {e}")
                spec_flux = np.zeros_like(f0)
                
            try:
                spec_rolloff = extract_spectral_rolloff(audio, sr)
                print(f"  ✓ Spectral rolloff: {len(spec_rolloff)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral rolloff for {file.name}: {e}")
                spec_rolloff = np.zeros_like(f0)
            
            # Features list
            features_list = [
                f0, 
                intensity_vals, 
                hnr_vals, 
                f1, 
                f2, 
                spec_centroid, 
                zcr, 
                spec_flux,
                spec_rolloff
            ]
            
            # Find minimum length among all features
            min_len = min(len(feat) for feat in features_list)
            
            # Make sure all features have the same length
            features_list = [feat[:min_len] for feat in features_list]
            
            # Add jitter and shimmer (constant values)
            jitter_array = np.full(min_len, jitter)
            shimmer_array = np.full(min_len, shimmer)
            
            # Stack all features
            features = np.column_stack(features_list + [jitter_array, shimmer_array])
            
            # Validate features
            if np.all(features[:, :5] == 0):  # Check F0, intensity, HNR, F1, F2
                print(f"Features for {file.name} are all zeros, skipping")
                logging.error(f"Features for {file.name} are all zeros, skipped")
                continue
                
            # Create DataFrame
            df = pd.DataFrame(
                features,
                columns=[
                    "f0", "intensity", "hnr", "formant1", "formant2", 
                    "spectral_centroid", "zero_crossing_rate", "spectral_flux",
                    "spectral_rolloff", "jitter", "shimmer"
                ]
            )
            
            # Save to TXT format (comma-separated with header)
            output_file = os.path.join(output_folder, file.stem + ".txt")
            df.to_csv(output_file, index=False)
            
            print(f"Successfully processed {file.name}")
            logging.info(f"Successfully processed {file.name} with all features")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            logging.error(f"Error processing {file.name}: {e}")
            continue

print("Feature extraction complete! See processing_log.txt for details.")
logging.info("Feature extraction complete!")
'''



import parselmouth
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import os
from pathlib import Path
import logging
from scipy import fft
from scipy.signal import stft

# Setup logging
logging.basicConfig(
    filename=r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features/processing_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directories
base_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio"
output_dir = r"/Users/rumenguin/Research/MERC/EmoReact/Audio_features"
folders = ["Train", "Test", "Validation"]
output_folders = ["Train_feat", "Test_feat", "Val_feat"]

# Parameters
sr = 16000  # Sample rate
time_step = 0.01  # 10 ms frame length
min_duration = 0.05  # Min 50 ms
energy_threshold = 1e-8  # Relaxed for EmoReact

# Voice Activity Detection (VAD)
def apply_vad(audio, sr):
    frame_len = int(0.02 * sr)  # 20 ms frames
    frame_shift = int(0.01 * sr)  # 10 ms shift
    energy = []
    for i in range(0, len(audio) - frame_len + 1, frame_shift):
        frame = audio[i:i + frame_len]
        energy.append(np.mean(frame ** 2))
    
    thresh = np.mean(energy) * 0.1
    speech_frames = [e > thresh for e in energy]
    
    speech_audio = np.zeros_like(audio)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * frame_shift
            end = min(start + frame_len, len(audio))
            speech_audio[start:end] = audio[start:end]
    
    # Trim zeros
    non_zero = np.where(speech_audio != 0)[0]
    if len(non_zero) > 0:
        speech_audio = speech_audio[non_zero[0]:non_zero[-1] + 1]
    if len(speech_audio) < sr * min_duration:
        return audio  # Fallback to original
    return speech_audio

# Helper functions for additional features
def extract_spectral_centroid(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract spectral centroid."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    centroids = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        if np.sum(frame**2) < 1e-10:  # Skip silent frames
            centroids.append(0)
            continue
            
        # Calculate FFT
        spectrum = np.abs(fft.rfft(frame * np.hamming(len(frame))))
        freqs = fft.rfftfreq(frame_size, 1/sr)
        
        # Calculate centroid
        if np.sum(spectrum) > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            centroids.append(centroid)
        else:
            centroids.append(0)
    
    return np.array(centroids)

def extract_zero_crossing_rate(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract zero crossing rate."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    zcr_values = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame).astype(int))))
        zcr = zero_crossings / (frame_size - 1)
        zcr_values.append(zcr)
    
    return np.array(zcr_values)

def extract_spectral_flux(audio, sr, frame_len=0.02, hop_len=0.01):
    """Extract spectral flux - measures the change in magnitude in the frequency spectrum."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    # Compute STFT
    frequencies, times, Zxx = stft(audio, sr, nperseg=frame_size, noverlap=frame_size-hop_size)
    
    # Calculate magnitude spectrum
    mag_spec = np.abs(Zxx)
    
    # Calculate flux (difference between consecutive spectra)
    flux = np.zeros(mag_spec.shape[1])
    for i in range(1, mag_spec.shape[1]):
        # Euclidean distance between consecutive spectra
        flux[i] = np.sqrt(np.sum((mag_spec[:, i] - mag_spec[:, i-1])**2))
    
    # Normalize
    if np.max(flux) > 0:
        flux = flux / np.max(flux)
    
    return flux

def extract_spectral_rolloff(audio, sr, frame_len=0.02, hop_len=0.01, rolloff_percent=0.85):
    """Extract spectral rolloff - frequency below which rolloff_percent of spectrum energy lies."""
    frame_size = int(frame_len * sr)
    hop_size = int(hop_len * sr)
    
    rolloffs = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        if np.sum(frame**2) < 1e-10:  # Skip silent frames
            rolloffs.append(0)
            continue
            
        # Calculate FFT
        spectrum = np.abs(fft.rfft(frame * np.hamming(len(frame))))
        freqs = fft.rfftfreq(frame_size, 1/sr)
        
        # Calculate rolloff
        if np.sum(spectrum) > 0:
            cumulative_sum = np.cumsum(spectrum)
            threshold = rolloff_percent * cumulative_sum[-1]
            # Find the frequency that corresponds to rolloff_percent of energy
            for j, energy in enumerate(cumulative_sum):
                if energy >= threshold:
                    rolloffs.append(freqs[j])
                    break
            else:
                rolloffs.append(0)
        else:
            rolloffs.append(0)
    
    return np.array(rolloffs)

# Helper function for jitter and shimmer extraction
def extract_jitter_shimmer(sound, file_name):
    try:
        # Create Pitch object optimized for voice analysis
        pitch = sound.to_pitch(0.75/75, 75, 600)
        
        # Create PointProcess object from Sound and Pitch
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        
        # Check if we have enough points for analysis
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        
        if num_points >= 3:
            # Calculate jitter using local method
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Calculate shimmer - this needs both Sound and PointProcess objects
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Additional check for unreasonable values
            if jitter > 1.0:  # Jitter values are typically very small (<0.1)
                logging.warning(f"File {file_name}: Very high jitter value ({jitter}), capping at 1.0")
                jitter = 1.0
                
            if shimmer > 1.0:  # Shimmer values typically <1.0
                logging.warning(f"File {file_name}: Very high shimmer value ({shimmer}), capping at 1.0")
                shimmer = 1.0
                
            logging.info(f"File {file_name}: Successfully extracted jitter ({jitter:.4f}) and shimmer ({shimmer:.4f})")
            return jitter, shimmer
        else:
            logging.warning(f"File {file_name}: Insufficient pulses for jitter/shimmer ({num_points} points found)")
            return 0.0, 0.0
    except Exception as e:
        logging.error(f"Failed to compute jitter/shimmer for {file_name}: {e}")
        return 0.0, 0.0

# Function to compute delta and delta-delta features
def compute_deltas(features, win_length=3):
    """
    Compute delta features (first derivative) and delta-delta features (second derivative)
    
    Args:
        features: numpy array of shape (n_frames, n_features)
        win_length: window length for delta computation (must be odd)
    
    Returns:
        deltas: numpy array of delta features (same shape as features)
        delta_deltas: numpy array of delta-delta features (same shape as features)
    """
    if win_length % 2 == 0:
        win_length += 1  # Make sure win_length is odd
    
    half_win = win_length // 2
    
    # Pad features to handle edge effects
    padded_features = np.pad(features, ((half_win, half_win), (0, 0)), mode='edge')
    
    deltas = np.zeros_like(features)
    
    # Compute delta (first derivative)
    for t in range(features.shape[0]):
        # Extract window
        window = padded_features[t:t+win_length]
        
        # Use regression formula
        numerator = np.sum([(i-half_win) * window[i] for i in range(win_length)], axis=0)
        denominator = np.sum([(i-half_win)**2 for i in range(win_length)])
        
        deltas[t] = numerator / denominator
    
    # Compute delta-delta (second derivative) by applying delta to delta
    delta_deltas = np.zeros_like(features)
    padded_deltas = np.pad(deltas, ((half_win, half_win), (0, 0)), mode='edge')
    
    for t in range(features.shape[0]):
        window = padded_deltas[t:t+win_length]
        numerator = np.sum([(i-half_win) * window[i] for i in range(win_length)], axis=0)
        denominator = np.sum([(i-half_win)**2 for i in range(win_length)])
        
        delta_deltas[t] = numerator / denominator
    
    return deltas, delta_deltas

# Function to compute statistical functionals
def compute_functionals(features, segment_length=50, overlap=25):
    """
    Compute statistical functionals (mean and std) over time segments
    
    Args:
        features: numpy array of shape (n_frames, n_features)
        segment_length: length of segments in frames
        overlap: overlap between segments in frames
    
    Returns:
        segment_means: mean values for each segment
        segment_stds: standard deviation values for each segment
    """
    n_frames, n_features = features.shape
    
    if n_frames < segment_length:
        # If too short, use whole file as one segment
        segment_means = np.mean(features, axis=0).reshape(1, -1)
        segment_stds = np.std(features, axis=0, ddof=1).reshape(1, -1)
        return segment_means, segment_stds
    
    hop_length = segment_length - overlap
    n_segments = max(1, (n_frames - segment_length) // hop_length + 1)
    
    segment_means = np.zeros((n_segments, n_features))
    segment_stds = np.zeros((n_segments, n_features))
    
    for i in range(n_segments):
        start = i * hop_length
        end = min(start + segment_length, n_frames)
        
        segment = features[start:end]
        segment_means[i] = np.mean(segment, axis=0)
        segment_stds[i] = np.std(segment, axis=0, ddof=1)
    
    return segment_means, segment_stds

# Process audio files
for i, folder in enumerate(folders):
    input_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, output_folders[i])
    os.makedirs(output_folder, exist_ok=True)
    
    for file in Path(input_folder).glob("*.wav"):
        try:
            print(f"Processing {file.name}")
            logging.info(f"Processing {file.name}")
            
            # Read audio
            audio, sr_file = sf.read(file)
            duration = len(audio) / sr_file
            
            # Validate duration
            if duration < min_duration:
                print(f"File {file.name} too short ({duration:.2f} s), skipping")
                logging.error(f"File {file.name} too short ({duration:.2f} s), skipped")
                continue
            
            # Limit duration (for very long files)
            if duration > 30:  # Cap at 30 seconds max
                audio = audio[:int(30 * sr_file)]
                logging.info(f"File {file.name} truncated to 30 seconds")
            
            # Convert to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16 kHz
            if sr_file != sr:
                audio = resample(audio, int(len(audio) * sr / sr_file))
            
            # Normalize
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9
            else:
                print(f"File {file.name} has zero amplitude, skipping")
                logging.error(f"File {file.name} has zero amplitude, skipped")
                continue
            
            # Apply VAD
            audio = apply_vad(audio, sr)
            if len(audio) < sr * min_duration:
                print(f"No significant speech in {file.name} after VAD, skipping")
                logging.error(f"No significant speech in {file.name} after VAD, skipped")
                continue
            
            # Compute energy
            energy = np.mean(audio ** 2)
            logging.info(f"File {file.name}: Duration {duration:.2f} s, Energy {energy:.6e}")
            if energy < energy_threshold:
                print(f"File {file.name} has low energy ({energy:.6e}), proceeding")
                logging.warning(f"File {file.name} has low energy ({energy:.6e}), proceeding")
            
            # Create Praat sound object
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Extract base features
            pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
            f0 = pitch.selected_array['frequency']
            
            intensity = snd.to_intensity(minimum_pitch=75, time_step=time_step)
            intensity_vals = intensity.values[0]
            
            hnr = snd.to_harmonicity(time_step=time_step, minimum_pitch=75)
            hnr_vals = hnr.values[0]
            
            formants = snd.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=5500)
            time_points = formants.xs()
            f1 = np.zeros(len(time_points))
            f2 = np.zeros(len(time_points))
            for t_idx, t in enumerate(time_points):
                try:
                    f1[t_idx] = formants.get_value_at_time(1, t, unit="HERTZ")
                    f2[t_idx] = formants.get_value_at_time(2, t, unit="HERTZ")
                except:
                    f1[t_idx] = np.nan
                    f2[t_idx] = np.nan
            # Replace NaNs with interpolated values or zeros
            f1 = pd.Series(f1).interpolate().fillna(0).values
            f2 = pd.Series(f2).interpolate().fillna(0).values
            
            # Extract jitter and shimmer using our helper function
            jitter, shimmer = extract_jitter_shimmer(snd, file.name)
            
            # Safe extraction of additional features with individual try-except blocks
            try:
                spec_centroid = extract_spectral_centroid(audio, sr)
                print(f"  ✓ Spectral centroid: {len(spec_centroid)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral centroid for {file.name}: {e}")
                spec_centroid = np.zeros_like(f0)
            
            try:
                zcr = extract_zero_crossing_rate(audio, sr)
                print(f"  ✓ Zero crossing rate: {len(zcr)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract zero crossing rate for {file.name}: {e}")
                zcr = np.zeros_like(f0)
            
            try:
                spec_flux = extract_spectral_flux(audio, sr)
                print(f"  ✓ Spectral flux: {len(spec_flux)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral flux for {file.name}: {e}")
                spec_flux = np.zeros_like(f0)
                
            try:
                spec_rolloff = extract_spectral_rolloff(audio, sr)
                print(f"  ✓ Spectral rolloff: {len(spec_rolloff)} frames")
            except Exception as e:
                logging.warning(f"Failed to extract spectral rolloff for {file.name}: {e}")
                spec_rolloff = np.zeros_like(f0)
            
            # Features list
            features_list = [
                f0, 
                intensity_vals, 
                hnr_vals, 
                f1, 
                f2, 
                spec_centroid, 
                zcr, 
                spec_flux,
                spec_rolloff
            ]
            
            # Find minimum length among all features
            min_len = min(len(feat) for feat in features_list)
            
            # Make sure all features have the same length
            features_list = [feat[:min_len] for feat in features_list]
            
            # Add jitter and shimmer (constant values)
            jitter_array = np.full(min_len, jitter)
            shimmer_array = np.full(min_len, shimmer)
            
            # Stack all features
            features = np.column_stack(features_list + [jitter_array, shimmer_array])
            
            # Validate features
            if np.all(features[:, :5] == 0):  # Check F0, intensity, HNR, F1, F2
                print(f"Features for {file.name} are all zeros, skipping")
                logging.error(f"Features for {file.name} are all zeros, skipped")
                continue
                
            # Create DataFrame with original features
            df_columns = [
                "f0", "intensity", "hnr", "formant1", "formant2", 
                "spectral_centroid", "zero_crossing_rate", "spectral_flux",
                "spectral_rolloff", "jitter", "shimmer"
            ]
            
            # Compute delta and delta-delta features
            try:
                print(f"Computing deltas for {file.name}")
                logging.info(f"Computing deltas for {file.name}")
                
                deltas, delta_deltas = compute_deltas(features)
                
                # Create column names for delta and delta-delta features
                delta_columns = [f"delta_{col}" for col in df_columns]
                delta_delta_columns = [f"delta_delta_{col}" for col in df_columns]
                
                # Combine all features
                all_features = np.column_stack([features, deltas, delta_deltas])
                all_columns = df_columns + delta_columns + delta_delta_columns
                
                print(f"  ✓ Delta features: {deltas.shape}")
                print(f"  ✓ Delta-delta features: {delta_deltas.shape}")
                
            except Exception as e:
                print(f"Error computing deltas for {file.name}: {e}")
                logging.error(f"Error computing deltas for {file.name}: {e}")
                # Fall back to original features if delta computation fails
                all_features = features
                all_columns = df_columns
            
            # Create DataFrame with all features
            df = pd.DataFrame(all_features, columns=all_columns)
            
            # Compute statistical functionals
            try:
                print(f"Computing statistical functionals for {file.name}")
                logging.info(f"Computing statistical functionals for {file.name}")
                
                segment_means, segment_stds = compute_functionals(all_features)
                
                # Create column names for functionals
                mean_columns = [f"mean_{col}" for col in all_columns]
                std_columns = [f"std_{col}" for col in all_columns]
                
                # Create DataFrame for functionals
                df_functionals = pd.DataFrame(
                    np.column_stack([segment_means, segment_stds]),
                    columns=mean_columns + std_columns
                )
                
                # Save functionals to separate file
                functionals_file = os.path.join(output_folder, file.stem + "_functionals.txt")
                df_functionals.to_csv(functionals_file, index=False)
                
                print(f"  ✓ Statistical functionals: {segment_means.shape[0]} segments")
                logging.info(f"Successfully extracted functionals for {file.name}")
                
            except Exception as e:
                print(f"Error computing functionals for {file.name}: {e}")
                logging.error(f"Error computing functionals for {file.name}: {e}")
            
            # Save frame-level features to TXT format (comma-separated with header)
            output_file = os.path.join(output_folder, file.stem + ".txt")
            df.to_csv(output_file, index=False)
            
            print(f"Successfully processed {file.name}")
            logging.info(f"Successfully processed {file.name} with all features")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            logging.error(f"Error processing {file.name}: {e}")
            continue

print("Feature extraction complete! See processing_log.txt for details.")
logging.info("Feature extraction complete!")