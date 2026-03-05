import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date
import glob
import librosa
import matplotlib as mpl
from scipy import signal

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Define paths
stimuli_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli/MovieTrailers"
music_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli-MaansiDesai/MovieTrailers/Separated_Speech_Music/Moises/music"
speech_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli-MaansiDesai/MovieTrailers/Separated_Speech_Music/Moises/speech"

save_dir = f"/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/frequency_comparison"
os.makedirs(save_dir, exist_ok=True)

colors = {
    'speech': '#c51b7d',
    'music': '#276419',
    'mixed': '#f4883c'
}

def read_audio_full(file_path):
    """Read entire audio file."""
    try:
        data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        if len(data) == 0:
            print(f"Warning: Empty file {os.path.basename(file_path)}")
            return None, None
            
        return sample_rate, data
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

def compute_spectrogram(data, sample_rate, n_fft=2048, hop_length=512):
    """Compute spectrogram in dB scale."""
    D = librosa.stft(data.astype(float), n_fft=n_fft, hop_length=hop_length)
    D_mag = np.abs(D)
    D_db = librosa.amplitude_to_db(D_mag, ref=np.max)
    return D_db

def plot_mean_power_comparison(speech_data, music_data, sample_rate, save_path):
    """Plot mean power comparison by averaging across time."""
    
    # Compute spectrograms
    n_fft = 2048
    hop_length = 512
    speech_db = compute_spectrogram(speech_data, sample_rate, n_fft, hop_length)
    music_db = compute_spectrogram(music_data, sample_rate, n_fft, hop_length)
    
    # Mean across time dimension (axis=1) to get average power at each frequency
    speech_mean_power = np.mean(speech_db, axis=1)
    music_mean_power = np.mean(music_db, axis=1)
    
    # Get frequency axis
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    freqs_khz = freqs / 1000  # Convert to kHz
    
    # Limit to 0-8 kHz
    freq_mask = freqs_khz <= 8
    freqs_khz = freqs_khz[freq_mask]
    speech_mean_power = speech_mean_power[freq_mask]
    music_mean_power = music_mean_power[freq_mask]
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Overlaid comparison
    ax.plot(freqs_khz, speech_mean_power, color=colors['speech'], 
            linewidth=2, label='Speech', alpha=0.8)
    ax.plot(freqs_khz, music_mean_power, color=colors['music'], 
            linewidth=2, label='Music', alpha=0.8)
    
    ax.set_xlabel('Frequency (kHz)', fontsize=13)
    ax.set_ylabel('Power (dB)', fontsize=13)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Get current date for filename
    today = date.today()
    date_str = f"{today.month}_{today.day}"
    
    plt.savefig(f"{save_path}_mean_power_{date_str}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}_mean_power_{date_str}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  → Saved mean power comparison plot with date: {date_str}")

# Get list of all original files
original_files = glob.glob(os.path.join(stimuli_dir, "*.wav"))

# Lists to store all concatenated data
all_speech_data = []
all_music_data = []
common_sr = None

print("Reading and concatenating audio files (full duration)...")
for orig_path in original_files:
    base_name = os.path.splitext(os.path.basename(orig_path))[0]
    music_path = os.path.join(music_dir, base_name + "_16kHz_Instrumental_mixed.wav")
    speech_path = os.path.join(speech_dir, base_name + "_16kHz_Vocals1_mixed.wav")
    
    if not os.path.exists(music_path) or not os.path.exists(speech_path):
        print(f"Skipping {base_name} - separated files not found")
        continue
    
    # Read full audio files
    sr_speech, data_speech = read_audio_full(speech_path)
    sr_music, data_music = read_audio_full(music_path)
    
    if data_speech is None or data_music is None:
        print(f"Skipping {base_name} - invalid audio data")
        continue
    
    # Store sample rate (should be same for all files)
    if common_sr is None:
        common_sr = sr_speech
    
    print(f"  Added {base_name} ({len(data_speech)/sr_speech:.2f}s) to concatenation")
    all_speech_data.append(data_speech)
    all_music_data.append(data_music)

# Concatenate all audio data in time dimension
if len(all_speech_data) == 0:
    print("No valid audio files found!")
else:
    print(f"\nConcatenating {len(all_speech_data)} audio files...")
    concatenated_speech = np.concatenate(all_speech_data)
    concatenated_music = np.concatenate(all_music_data)
    
    print(f"Total concatenated length: {len(concatenated_speech)/common_sr:.2f} seconds")
    
    # Save path
    save_base = os.path.join(save_dir, "all_files_full")
    
    # Create mean power comparison plot
    print("Computing mean power spectrum...")
    plot_mean_power_comparison(concatenated_speech, concatenated_music, common_sr, save_base)
    
    print(f"\nMean power comparison plot saved to: {save_dir}")