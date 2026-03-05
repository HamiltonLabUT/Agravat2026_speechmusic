from ECoG_phn_alignment_tools import *
from create_h5_funcs import *
import pandas as pd
from subjects import *


def ch_intersect(data_dir, block_list):
    ch_names = []
    for t in block_list:
        raw = load_raw_ECoG(t, data_dir, 'high_gamma')
        cleaned = [ch.replace('EEG ', '') if 'EEG ' in ch else ch for ch in raw.info['ch_names']]
        ch_names.append(cleaned)

    picks = set.intersection(*map(set, ch_names))
    print(f'{len(picks)} shared channels in {data_dir}')
    return picks


def create_h5(user, block_list, subject, data_dir, nat_sound_event_dir, textgrid_dir, stimulus_class, picks, old_tch_subjs, band):
    wav_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/Stimuli/MovieTrailers'
    h5_name = f'{data_dir}/{subject}_ECoG_matrix.hf5' if band == 'high_gamma' else f'{data_dir}/{subject}_ECoG_matrix_{band}.hf5'

    with h5py.File(h5_name, 'a') as g:
        epochs_list = {stimulus_class: {}}

        for block in block_list:
            if subject in old_tch_subjs:
                event_file = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe_new.csv')
                txt_file = f'{data_dir}/{block}/{block}_sentence-eve_new.txt'
            else:
                event_file = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe.csv')
                txt_file = f'{data_dir}/{block}/{block}_sentence-eve.txt'

            evs = np.loadtxt(txt_file, dtype='f', usecols=(0, 1, 2))
            evs = np.atleast_2d(evs).astype(int)

            raw = load_raw_ECoG(block, data_dir, band)
            channels = [ch.replace('EEG ', '') if 'EEG ' in ch else ch for ch in raw.info['ch_names']]
            bad_chs = [ch for ch in channels if ch not in picks]

            if subject == 'TCH45':
                ch_names_stripped = [ch.split()[-1] for ch in raw.info['ch_names']]
                bad_indices = [i for i, ch in enumerate(ch_names_stripped) if ch in bad_chs]
                bad_chs = [raw.info['ch_names'][i] for i in bad_indices]

            raw.drop_channels(bad_chs)
            new_fs = int(raw.info['sfreq'])
            channel_names = raw.info['ch_names']
            print(f'Block: {block} | Shape: {raw.get_data().shape}')

            for event in evs:
                wav_name = event_file['name'][event_file['event_id'] == event[2]].values[0] + '.wav'

                if wav_name in ('TIMIT_tone.wav', 'TIMIT_tone_stereo.wav'):
                    continue

                print(f'Event: {event} | Wav: {wav_name}')
                epochs = get_event_epoch(raw, [event], event[2], trailers=True)

                if wav_name not in epochs_list[stimulus_class]:
                    epochs_list[stimulus_class][wav_name] = []
                epochs_list[stimulus_class][wav_name].append(epochs)

                binary_feat_mat, binary_phn_mat = binary_phn_mat_stim(user, stimulus_class, block, wav_name.replace('.wav', ''), epochs, data_dir)

                try:
                    g.create_dataset(f'{stimulus_class}/{wav_name}/stim/phn_feat_timings', data=np.array(binary_feat_mat, dtype=float))
                    g.create_dataset(f'{stimulus_class}/{wav_name}/stim/phn_timings', data=np.array(binary_phn_mat, dtype=float))
                except:
                    print('phn_timings already exists')

                if f'{stimulus_class}/{wav_name}/stim/spec' not in g:
                    envelope = make_envelopes(wav_dir, wav_name, new_fs, epochs, pad_next_pow2=True)
                    mel_spec, freqs = stimuli_mel_spec(wav_dir, wav_name, new_fs=new_fs)
                    pitch_values = get_meanF0s_v2(fileName=os.path.join(wav_dir, wav_name))
                    binned_pitch_edges = get_bin_edges_percent_range(pitch_values)
                    binned_pitches = get_pitch_matrix(scipy.signal.resample(pitch_values, binary_feat_mat.shape[1]), binned_pitch_edges)

                    try:
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/spec', data=np.array(mel_spec, dtype=float))
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/freqs', data=np.array(freqs, dtype=float))
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/pitches', data=np.array(pitch_values, dtype=float))
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/envelope', data=np.array(envelope, dtype=float))
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/binned_pitches', data=np.array(binned_pitches, dtype=float))
                        g.create_dataset(f'{stimulus_class}/{wav_name}/stim/binned_edges', data=np.array(binned_pitch_edges, dtype=float))
                    except:
                        print('Stim features already exist')

                # if stimulus_class == 'MovieTrailers':
                #     if wav_name in ('the-lego-ninjago-movie-trailer-2_a720p.wav', 'clip_rep.wav'):
                #         continue
                #     if wav_name != 'frozen-tlr2_a720p.wav':
                #         full_cat_mat = trailer_soundCat_matrix(nat_sound_event_dir, wav_name, evs, epochs, show_fig=True)
                #         sc = scene_cut_feature(wav_name, textgrid_dir, fs=new_fs)
                #         try:
                #             g.create_dataset(f'{stimulus_class}/{wav_name}/stim/nat_sound', data=np.array(full_cat_mat, dtype=float))
                #             g.create_dataset(f'{stimulus_class}/{wav_name}/stim/scene_cut', data=np.array(sc, dtype=float))
                #         except:
                #             print('Nat sound / scene cut already exists')
                #     else:
                #         print('Skipping frozen (nat sound not available)')

        for wav_name, ep_list in epochs_list[stimulus_class].items():
            if len(ep_list) > 1 and np.shape(ep_list[0]) != np.shape(ep_list[1]):
                ep_list[0] = scipy.signal.resample(ep_list[0], np.shape(ep_list[0])[2], axis=2)

            epochs_resized = ep_list[0]
            print(f'Saving epochs for {wav_name}: {epochs_resized.shape}')
            try:
                g.create_dataset(f'{stimulus_class}/{wav_name}/resp/epochs', data=epochs_resized)
            except:
                print(f'Epochs already exist for {wav_name}')

        np.savetxt(f'{data_dir}/{subject}_channelnames_{band}.txt', channel_names, delimiter=' ', fmt='%s')


if __name__ == "__main__":
    user = 'rajviagravat'
    stimulus_class = 'MovieTrailers'  # or 'TIMIT'
    band = 'high_gamma'

    old_tch_subjs = ['TCH13', 'TCH14', 'TCH15', 'TCH18', 'TCH19', 'TCH20', 'TCH21', 'TCH22', 'TCH23', 'TCH25', 'TCH26']
    all_subjs, all_subjs_mt, all_subjs_timit = seeg_subjs()

    nat_sound_event_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/Stimuli/natural_sounds_mt'
    textgrid_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/trailer_AV/textgrids/scene_cut_textGrids/'

    # Define subjects, their site, and blocks here 
    subjects = {
        'TCH65': {'site': 'TCH', 'blocks': ['TCH65_B8']}
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
        create_h5(user, block_list, subject, data_dir, nat_sound_event_dir, textgrid_dir, stimulus_class, picks, old_tch_subjs, band)
        print(f'Done: {subject} | {stimulus_class}')