import sys
import os
import traceback
import logging
import h5py
import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

preproc_dir = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/preproc'
ridge_dir   = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/ridge/'
sys.path.append(preproc_dir)
sys.path.append(ridge_dir)

from utils import make_delayed, save_table_file
from ridge_ import bootstrap_ridge

zs = lambda x: (x - x[np.isnan(x) == False].mean(0)) / x[np.isnan(x) == False].std(0)


def loadEEGh5(subject, stimulus_class, data_dir, resp_mean=True):
    stim_dict = {}
    resp_dict = {}

    with h5py.File(f'{data_dir}/sub-{subject}/{subject}_ECoG_speechmusic.hf5', 'r') as fh:
        all_stim = list(fh[f'/{stimulus_class}'].keys())
        print(f"All stimuli: {all_stim}")

        for wav_name in all_stim:
            stim_dict[wav_name] = []
            resp_dict[wav_name] = []
            try:
                specs  = fh[f'/{stimulus_class}/{wav_name}/stim/spec'][:]
                ntimes = specs.shape[1]
                specs  = scipy.signal.resample(specs, ntimes, axis=1)
                specs  = zs(specs)
                stim_dict[wav_name].append(specs)

                freqs = fh[f'/{stimulus_class}/{wav_name}/stim/freqs'][:]

                epochs_data = fh[f'/{stimulus_class}/{wav_name}/resp/epochs'][:]
                if resp_mean:
                    epochs_data = epochs_data.mean(0)
                    epochs_data = scipy.signal.resample(epochs_data.T, ntimes).T
                else:
                    epochs_data = scipy.signal.resample(epochs_data, ntimes, axis=2)
                resp_dict[wav_name].append(epochs_data)

            except Exception:
                traceback.print_exc()

    return resp_dict, stim_dict, freqs


def strf_spec_refit(subject, stimulus_class, data_dir, test_set,
                    fs=100.0, delay_min=0.0, delay_max=1.0):
    output_dir = f'{data_dir}/sub-{subject}/strfs_speechmusic'
    os.makedirs(output_dir, exist_ok=True)

    resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, resp_mean=True)

    all_stimuli  = [k for k in stim_dict.keys() if len(resp_dict[k]) > 0]
    training_set = [s for s in all_stimuli if s not in test_set]

    print(f"Training set ({len(training_set)}): {training_set}")
    print(f"Test set ({len(test_set)}): {test_set}")

    tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
    vResp = np.hstack([resp_dict[r][0] for r in test_set]).T

    tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
    vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))

    for arr in [tStim_temp, vStim_temp]:
        arr[np.isnan(arr)] = 0

    delays = np.arange(np.floor(delay_min * fs), np.ceil(delay_max * fs), dtype=int)
    tStim  = make_delayed(tStim_temp, delays)
    vStim  = make_delayed(vStim_temp, delays)

    print(f"tStim: {tStim.shape}, tResp: {tResp.shape}")
    print(f"vStim: {vStim.shape}, vResp: {vResp.shape}")

    alphas   = np.hstack((0, np.logspace(2, 8, 20)))
    chunklen = int(len(delays) * 3)
    nchunks  = int(np.floor(0.2 * tStim.shape[0] / chunklen))

    logging.basicConfig(level=logging.DEBUG)

    wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(
        tStim, tResp, vStim, vResp,
        alphas, nboots=10, chunklen=chunklen, nchunks=nchunks,
        use_corr=True, single_alpha=False, use_svd=False,
        corrmin=0.05, joined=None
    )

    channel_names_file = f'{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt'
    channel_names = np.loadtxt(channel_names_file, usecols=(0), dtype=str)

    train_inds = np.array([i for i, s in enumerate(all_stimuli) if s in training_set])
    val_inds   = np.array([i for i, s in enumerate(all_stimuli) if s in test_set])

    strf_file = f'{output_dir}/STRF_by_spec_MT_{stimulus_class}.hf5'
    print(f"Saving to {strf_file}")

    with h5py.File(strf_file, 'w') as f:
        f.create_dataset('/ch_names',      data=np.array(channel_names, dtype=h5py.string_dtype()))
        f.create_dataset('/wts_mt',        data=wt)
        f.create_dataset('/corrs_mt',      data=corrs)
        f.create_dataset('/valphas_mt',    data=valphas)
        f.create_dataset('/allRcorrs_mt',  data=allRcorrs)
        f.create_dataset('/train_inds_mt', data=train_inds)
        f.create_dataset('/val_inds_mt',   data=val_inds)
        f.create_dataset('/delays_mt',     data=delays)
        f.create_dataset('/freqs',         data=freqs)
        f.create_dataset('/test_set',      data=np.array(test_set,     dtype=h5py.string_dtype()))
        f.create_dataset('/training_set',  data=np.array(training_set, dtype=h5py.string_dtype()))

    print("Done.")
    return wt, corrs, valphas, allRcorrs


if __name__ == "__main__":
    user = 'rajviagravat'

    subjects = {
        'TCH65': ['pandas-trailer-2_a720p.wav']
    }

    for subject, test_set in subjects.items():
        data_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/{"TCH_ECoG" if subject.startswith("TCH") else "ECoG_Backup"}'
        for stimulus_class in ['speech', 'music']:
            print(f"\nProcessing {subject} - {stimulus_class}")
            wt, corrs, valphas, allRcorrs = strf_spec_refit(
                subject, stimulus_class, data_dir, test_set,
                fs=100.0, delay_min=0.0, delay_max=1.0
            )