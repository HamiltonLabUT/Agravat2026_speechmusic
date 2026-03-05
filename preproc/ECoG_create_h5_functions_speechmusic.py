# ECoG_create_h5_speechmusic.py
# Creates h5 files for speech-separated and music-separated models.
# Post-march 2025: takes onset/offset times from event_times_dataframe csv. - RA 3/14/2025

from ECoG_phn_alignment_tools import *
from create_h5_funcs import *
import pandas as pd
import numpy as np
import h5py
import scipy.signal


def ch_intersect(data_dir, block_list):
    ch_names = []
    for t in block_list:
        raw = load_raw_ECoG(t, data_dir, 'high_gamma')
        print(t)
        print(raw.info['ch_names'])
        ch_names.append(raw.info['ch_names'])

    picks = set.intersection(*map(set, ch_names))
    print(f'{len(picks)} shared channels in {data_dir}')
    return picks


def create_h5(user, block_list, subject, data_dir, stimulus_class, picks):
    wav_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/Stimuli/MovieTrailersSplit'

    with h5py.File(f'{data_dir}/{subject}_ECoG_speechmusic.hf5', 'a') as g:
        epochs_list = {s: {} for s in stimulus_class}

        for s in stimulus_class:
            print(f'Stimulus type: {s}')

            if s == 'speech':
                ext = '_16kHz_Vocals1_mixed'
            elif s == 'music':
                ext = '_16kHz_Instrumental_mixed'
            else:
                raise ValueError(f'Unknown stimulus class: {s}. Expected "speech" or "music".')

            for block in block_list:
                event_file = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe.csv')
                raw = load_raw_ECoG(block, data_dir, 'high_gamma')
                bad_chs = [ch for ch in raw.info['ch_names'] if ch not in picks]
                new_fs = int(raw.info['sfreq'])
                raw.drop_channels(bad_chs)

                channel_names = raw.info['ch_names']
                np.savetxt(f'{data_dir}/{subject}_channelnames_speech_music.txt', channel_names, delimiter=' ', fmt='%s')

                if '/channel_names' not in g:
                    g.create_dataset('/channel_names', data=np.array(list(picks), dtype=h5py.string_dtype()))

                print(f'Block: {block} | Shape: {raw.get_data().shape}')

                for _, event_row in event_file.iterrows():
                    onset_sample = int(event_row['onset_time'] * new_fs)
                    offset_sample = int(event_row['offset_time'] * new_fs)
                    event_id = event_row['event_id']
                    wav_name = event_row['name'] + '.wav'
                    mixed_name = event_row['name'] + ext + '.wav'

                    print(f'Event: {event_row["name"]} | Mixed file: {mixed_name}')

                    epochs = get_event_epoch(raw, [(onset_sample, offset_sample, event_id)], event_id, trailers=True)

                    if wav_name not in epochs_list[s]:
                        epochs_list[s][wav_name] = []
                    epochs_list[s][wav_name].append(epochs)

                    if f'{s}/{wav_name}/stim/spec' not in g:
                        envelope = make_envelopes(wav_dir, mixed_name, new_fs, epochs, pad_next_pow2=True)
                        mel_spec, freqs = stimuli_mel_spec(wav_dir, mixed_name, new_fs=new_fs)
                        pitch_values = get_meanF0s_v2(fileName=os.path.join(wav_dir, mixed_name), f0min=50, f0max=8000)
                        binned_pitch_edges = get_bin_edges_percent_range(pitch_values)
                        binned_pitches = get_pitch_matrix(scipy.signal.resample(pitch_values, pitch_values.shape[0]), binned_pitch_edges)

                        try:
                            g.create_dataset(f'{s}/{wav_name}/stim/spec', data=np.array(mel_spec, dtype=float))
                            g.create_dataset(f'{s}/{wav_name}/stim/freqs', data=np.array(freqs, dtype=float))
                            g.create_dataset(f'{s}/{wav_name}/stim/pitches', data=np.array(pitch_values, dtype=float))
                            g.create_dataset(f'{s}/{wav_name}/stim/envelope', data=np.array(envelope, dtype=float))
                            g.create_dataset(f'{s}/{wav_name}/stim/binned_pitches', data=np.array(binned_pitches, dtype=float))
                            g.create_dataset(f'{s}/{wav_name}/stim/binned_edges', data=np.array(binned_pitch_edges, dtype=float))
                        except:
                            print(f'Stim features already exist for {wav_name}')

            for wav_name, ep_list in epochs_list[s].items():
                if len(ep_list) > 1 and np.shape(ep_list[0]) != np.shape(ep_list[1]):
                    ntimes = np.shape(ep_list[0])[2]
                    ep_list[0] = scipy.signal.resample(ep_list[0], ntimes, axis=2)

                epochs_resized = ep_list[0]
                print(f'Saving epochs for {wav_name}: {epochs_resized.shape}')
                try:
                    g.create_dataset(f'{s}/{wav_name}/resp/epochs', data=epochs_resized)
                except:
                    print(f'Epochs already exist for {wav_name}')


if __name__ == "__main__":
    user = 'rajviagravat'
    stimulus_class = ['speech', 'music']  # both separated models

    # Define subjects, their site, and blocks here 
    subjects = {
        'TCH43': {'site': 'TCH', 'blocks': ['TCH43_B1','TCH43_B3','TCH43_B5','TCH43_B8','TCH43_B10','TCH43_B14','TCH43_B16','TCH43_B18']},
        'TCH45': {'site': 'TCH', 'blocks': ['TCH45_B1','TCH45_B6','TCH45_B8','TCH45_B13','TCH45_B15','TCH45_B18','TCH45_B19','TCH45_B22','TCH45_B24','TCH45_B27','TCH45_B29']},
    }

    for subject, info in subjects.items():
        site, block_list = info['site'], info['blocks']

        if site == 'TCH':
            base_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/{site}_ECoG'
        elif site == 'DCMC':
            base_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/ECoG_Backup'
        else:
            raise ValueError(f'Unknown site: {site}')

        data_dir = f'{base_dir}/sub-{subject}'
        print(f'\n{"="*50}\nProcessing {subject} | Blocks: {block_list}\n{"="*50}')

        picks = ch_intersect(data_dir, block_list)
        create_h5(user, block_list, subject, data_dir, stimulus_class, picks)
        print(f'Done: {subject}')