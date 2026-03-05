import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date
import glob
import librosa
import librosa.display
import matplotlib as mpl

plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42

# Define paths
stimuli_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli/MovieTrailers"
music_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli/MovieTrailersSplit"
speech_dir = "/Users/rajviagravat/Library/CloudStorage/Box-Box/Stimuli/MovieTrailersSplit"

save_dir = f"/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/waveform_DNN/{date.today()}"
os.makedirs(save_dir, exist_ok=True)

# Define time segment to plot (in seconds)
start_time = 3
end_time = 16

colors = {
    'speech': '#c51b7d',
    'music': '#276419',
    'both': '#f4883c'
}

def read_audio(file_path, start_sec=None, end_sec=None):
    """Read audio file and extract specific time segment if requested."""
    try:
        data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        if len(data) == 0:
            print(f"Warning: Empty file {os.path.basename(file_path)}")
            return None, None
            
        data = data / np.max(np.abs(data))  # Normalize to [-1, 1]
        
        if start_sec is not None and end_sec is not None:
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            if start_idx >= len(data):
                print(f"Warning: File {os.path.basename(file_path)} is shorter than {start_sec} seconds")
                return None, None
            if end_idx > len(data):
                print(f"Warning: File {os.path.basename(file_path)} is shorter than {end_sec} seconds")
                end_idx = len(data)
            data = data[start_idx:end_idx]
            if len(data) == 0:
                print(f"Warning: No data in selected segment for {os.path.basename(file_path)}")
                return None, None
        
        return sample_rate, data
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

def plot_waveform_spectrogram(ax_wave, ax_spec, data, sample_rate, color, time_offset=0):
    """Plot waveform and spectrogram with proper time axis."""
    # Waveform
    duration = len(data) / float(sample_rate)
    time = np.linspace(time_offset, time_offset + duration, len(data))
    ax_wave.plot(time, data, color=color, linewidth=0.7)
    ax_wave.set_xlim([time_offset, time_offset + duration])
    ax_wave.set_ylim([-1.2, 1.2])  # For normalized signals
    ax_wave.axis('off')
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data.astype(float), n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='linear', ax=ax_spec, cmap='inferno')
    ax_spec.set_xlim([0, duration])
    ax_spec.set_xticks(np.linspace(0, duration, 4))
    ax_spec.set_xticklabels([f"{time_offset+t:.0f}" for t in np.linspace(0, duration, 4)], fontsize=10)
    ax_spec.set_ylim([0, 8000])
    ax_spec.set_yticks([0, 4000, 8000])  
    ax_spec.tick_params(axis='y', labelsize=8)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_spec.set_xlabel("Time (seconds)", fontsize=10)

# Get list of all original files
original_files = glob.glob(os.path.join(stimuli_dir, "*.wav"))

for orig_path in original_files:
    base_name = os.path.splitext(os.path.basename(orig_path))[0]
    music_path = os.path.join(music_dir, base_name + "_16kHz_Instrumental_mixed.wav")
    speech_path = os.path.join(speech_dir, base_name + "_16kHz_Vocals1_mixed.wav")
    
    if not os.path.exists(music_path) or not os.path.exists(speech_path):
        print(f"Skipping {base_name} - separated files not found")
        continue
    
    # Read and check all files first
    sr_orig, data_orig = read_audio(orig_path, start_time, end_time)
    sr_speech, data_speech = read_audio(speech_path, start_time, end_time)
    sr_music, data_music = read_audio(music_path, start_time, end_time)
    
    # Check if any of the data is None (invalid/short file)
    if data_orig is None or data_speech is None or data_music is None:
        print(f"Skipping {base_name} - one or more files are too short or invalid")
        continue
    
    # Create figure
    fig = plt.figure(figsize=(3, 7))
    
    # Gridspec layout
    gs = fig.add_gridspec(11, 1, height_ratios=[
        3,    # Speech+Music waveform
        3,    # Speech+Music spectrogram
        0.5,  # Padding
        0.5,  # Padding
        3,    # Speech-only waveform
        3,    # Speech-only spectrogram
        0.5,  # Padding
        0.5,  # Padding
        3,    # Music-only waveform
        3,    # Music-only spectrogram
        0     # No final padding row
    ])
    
    # Create axes
    axes = [
        fig.add_subplot(gs[0, 0]),  # Speech+Music waveform
        fig.add_subplot(gs[1, 0]),  # Speech+Music spectrogram
        fig.add_subplot(gs[4, 0]),  # Speech-only waveform
        fig.add_subplot(gs[5, 0]),  # Speech-only spectrogram
        fig.add_subplot(gs[8, 0]),  # Music-only waveform
        fig.add_subplot(gs[9, 0])   # Music-only spectrogram
    ]
    
    # Plot speech+music (original)
    plot_waveform_spectrogram(axes[0], axes[1], data_orig, sr_orig, colors['both'], time_offset=start_time)
    axes[0].set_title(f"Speech+Music ({start_time}-{end_time}s)", loc='left', fontsize=10)
    
    # Plot speech-only
    plot_waveform_spectrogram(axes[2], axes[3], data_speech, sr_speech, colors['speech'], time_offset=start_time)
    axes[2].set_title(f"Speech Only ({start_time}-{end_time}s)", loc='left', fontsize=10)
    
    # Plot music-only
    plot_waveform_spectrogram(axes[4], axes[5], data_music, sr_music, colors['music'], time_offset=start_time)
    axes[4].set_title(f"Music Only ({start_time}-{end_time}s)", loc='left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(h_pad=2.0)
    plt.subplots_adjust(hspace=0.6)
    
    save_base = os.path.join(save_dir, f"waveform_spectrogram_{base_name}_{start_time}-{end_time}s")
    plt.savefig(f"{save_base}.png", dpi=400, bbox_inches='tight')
    plt.savefig(f"{save_base}.pdf", dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_base}")

print(f"All visualizations saved to: {save_dir}")