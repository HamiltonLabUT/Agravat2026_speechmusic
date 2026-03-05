import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import wavfile
from scipy.signal import spectrogram, resample
import librosa
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

base_dir = '/Users/rajviagravat'
data_path = os.path.join(
    base_dir,
    'Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_10_15.csv'
)

today = datetime.now().strftime('%Y_%m_%d')

# Separate output folder
output_dir = os.path.join(base_dir, 'Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/sliding_corrs/single_electrode_TCH06_RTG2')
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, f'TCH06_RTG2_all_models_{today}.pdf')

# Load and filter CSV
df = pd.read_csv(data_path)

filtered_df = df[
    (df['subj_id'] == 'TCH06') &
    (df['channelnames'] == 'RTG2')
]

print(f"Found {len(filtered_df)} matching electrode(s)")

if filtered_df.empty:
    raise ValueError("No matching electrode found for TCH06 RTG2")

# --- Helper functions ---
def make_delayed(data, delays):
    delayed_data = [np.roll(data, shift=d, axis=0) for d in delays]
    delayed_data = np.stack(delayed_data, axis=-1)
    delayed_data[:delays.max(), :, :] = 0
    return delayed_data.reshape(data.shape[0], -1)

def zs(x):
    return (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)

def generate_spectrogram_from_wav(wav_path, target_length, n_freqs=80):
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(float)
    audio /= np.max(np.abs(audio))

    nperseg = int(sr * 0.02)
    noverlap = int(nperseg * 0.5)

    _, _, Sxx = spectrogram(audio, sr, nperseg=nperseg, noverlap=noverlap)

    if Sxx.shape[0] > n_freqs:
        Sxx = Sxx[:n_freqs]
    elif Sxx.shape[0] < n_freqs:
        Sxx = np.pad(Sxx, ((0, n_freqs - Sxx.shape[0]), (0, 0)))

    Sxx = np.log(Sxx + 1e-10)
    Sxx = resample(Sxx, target_length, axis=1)
    return Sxx

def load_audio_waveform(wav_path, start_sec, duration_sec):
    data, sr = librosa.load(wav_path, sr=None, mono=True)
    data /= np.max(np.abs(data))

    start_idx = int(start_sec * sr)
    end_idx = int((start_sec + duration_sec) * sr)
    waveform = data[start_idx:end_idx]

    time_axis = np.linspace(start_sec, start_sec + len(waveform) / sr, len(waveform))
    return waveform, time_axis

sample_rate = 100
delays = np.arange(100)

model_types = ['speech_only', 'music_only', 'speech_music', 'stacked']
model_colors = {
    'speech_only': '#c51b7d',
    'music_only': '#276419',
    'speech_music': '#f4883c',
    'stacked': '#6C3BAA'
}

stimuli_dir = os.path.join(base_dir, 'Library/CloudStorage/Box-Box/Stimuli/MovieTrailersSplit')
original_stimuli_dir = os.path.join(base_dir, 'Library/CloudStorage/Box-Box/Stimuli/MovieTrailers')

# Plot
with PdfPages(pdf_path) as pdf:

    row = filtered_df.iloc[0]

    subj_id = row['subj_id']
    channel_name = row['channelnames']
    chan_idx = int(row['channel'])
    short_anat = row.get('short_anat', 'Unknown')

    ecog_paths = [
        f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{subj_id}_ECoG_matrix.hf5',
        f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{subj_id}_ECoG_matrix.hf5'
    ]

    ecog_path = next(p for p in ecog_paths if os.path.exists(p))

    with h5py.File(ecog_path, 'r') as f:
        trailer_name = list(f['MovieTrailers'].keys())[0]

        vResp = zs(f[f'MovieTrailers/{trailer_name}/resp/epochs'][:].mean(0).T)
        vstim = zs(f[f'MovieTrailers/{trailer_name}/stim/spec'][:])

        target_length = vstim.shape[1]
        vPred_dict = {}

        # stacked spectrogram 
        base_filename = trailer_name.replace('.wav', '')
        speech_wav = os.path.join(stimuli_dir, f'{base_filename}_16kHz_Vocals1_mixed.wav')
        music_wav = os.path.join(stimuli_dir, f'{base_filename}_16kHz_Instrumental_mixed.wav')

        stacked_vstim = None
        if os.path.exists(speech_wav) and os.path.exists(music_wav):
            speech_spec = zs(generate_spectrogram_from_wav(speech_wav, target_length))
            music_spec = zs(generate_spectrogram_from_wav(music_wav, target_length))
            stacked_vstim = np.vstack([speech_spec, music_spec])

        # load models 
        for model_type in model_types:

            if model_type == 'stacked' and stacked_vstim is None:
                continue

            if model_type == 'speech_only':
                fname = 'STRF_by_spec_MT_speech.hf5'
                folder = 'strfs_speechmusic'
                wts_key = 'wts_mt'
            elif model_type == 'music_only':
                fname = 'STRF_by_spec_MT_music.hf5'
                folder = 'strfs_speechmusic'
                wts_key = 'wts_mt'
            elif model_type == 'stacked':
                fname = 'STRF_by_stacked_speechmusic_spec.hf5'
                folder = 'strfs_stacked_speechmusic'
                wts_key = 'wts'
            else:
                fname = 'STRF_by_spec_MT.hf5'
                folder = 'strfs_og'
                wts_key = 'wts'

            strf_path = next(
                p for p in [
                    f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{folder}/{fname}',
                    f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{folder}/{fname}'
                ]
                if os.path.exists(p)
            )

            with h5py.File(strf_path, 'r') as f_strf:
                wts = f_strf[wts_key][:]

                stim_use = stacked_vstim.T if model_type == 'stacked' else vstim.T
                stim_del = make_delayed(zs(stim_use), delays)

                vPred = zs(stim_del @ wts)
                vPred_dict[model_type] = vPred

        # 3 second window 
        window_size = 3 * sample_rate
        time_axis = np.arange(vResp.shape[0]) / sample_rate
        best_start = vResp.shape[0] // 2

        zoom_start = best_start
        zoom_end = zoom_start + window_size
        zoom_time = time_axis[zoom_start:zoom_end]

        # audio 
        audio_wave, audio_time = load_audio_waveform(
            os.path.join(original_stimuli_dir, trailer_name),
            zoom_time[0],
            zoom_time[-1] - zoom_time[0]
        )

        # plot 
        fig = plt.figure(figsize=(6, 5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.15)

        ax_wave = fig.add_subplot(gs[0])
        ax_pred = fig.add_subplot(gs[1])

        ax_wave.plot(audio_time, audio_wave, 'k', lw=0.5)
        ax_wave.axis('off')

        ax_pred.plot(zoom_time, vResp[zoom_start:zoom_end, chan_idx], 'k', lw=1.5)

        for m in ['speech_only', 'speech_music', 'music_only', 'stacked']:
            if m in vPred_dict:
                ax_pred.plot(
                    zoom_time,
                    vPred_dict[m][zoom_start:zoom_end, chan_idx],
                    color=model_colors[m],
                    lw=1.5,
                    label=m.replace('_', ' ').title()
                )

        ax_pred.set_ylim(-3, 2)
        ax_pred.set_yticks(np.arange(-3, 3, 1))

        ax_pred.xaxis.set_major_locator(MultipleLocator(1))
        ax_pred.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax_pred.set_xlabel('Time (s)')
        ax_pred.set_ylabel('Amplitude (A.U.)')

        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.tight_layout()
        plt.subplots_adjust(right=0.75)

        pdf.savefig(fig)
        fig.savefig(os.path.join(output_dir, 'TCH06_RTG2_all_models.pdf'), dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"Saved figure to {output_dir}")
