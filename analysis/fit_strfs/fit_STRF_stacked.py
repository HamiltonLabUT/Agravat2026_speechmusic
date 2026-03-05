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

# Paths
preproc_dir = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/preproc'
ridge_dir = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/ridge/'
sys.path.append(preproc_dir)
sys.path.append(ridge_dir)

from utils import make_delayed, save_table_file
from ridge_ import bootstrap_ridge

zs = lambda x: (x - x[np.isnan(x) == False].mean(0)) / x[np.isnan(x) == False].std(0)


# Data loading
def loadEEGh5_stacked(subject, data_dir, resp_mean=True):
    stim_dict_speech = {}
    stim_dict_music  = {}
    resp_dict        = {}

    with h5py.File(f'{data_dir}/sub-{subject}/{subject}_ECoG_speechmusic.hf5', 'r') as fh:
        all_stim = list(fh['/speech'].keys())
        print(f"All stimuli: {all_stim}")

        for wav_name in all_stim:
            stim_dict_speech[wav_name] = []
            stim_dict_music[wav_name]  = []
            resp_dict[wav_name]        = []
            try:
                specs_speech = fh[f'/speech/{wav_name}/stim/spec'][:]
                ntimes       = specs_speech.shape[1]
                specs_speech = scipy.signal.resample(specs_speech, ntimes, axis=1)
                specs_speech = zs(specs_speech)
                stim_dict_speech[wav_name].append(specs_speech)

                specs_music = fh[f'/music/{wav_name}/stim/spec'][:]
                specs_music = scipy.signal.resample(specs_music, ntimes, axis=1)
                specs_music = zs(specs_music)
                stim_dict_music[wav_name].append(specs_music)

                freqs = fh[f'/speech/{wav_name}/stim/freqs'][:]

                epochs_data = fh[f'/speech/{wav_name}/resp/epochs'][:]
                if resp_mean:
                    epochs_data = epochs_data.mean(0)
                    epochs_data = scipy.signal.resample(epochs_data.T, ntimes).T
                else:
                    epochs_data = scipy.signal.resample(epochs_data, ntimes, axis=2)
                resp_dict[wav_name].append(epochs_data)

            except Exception:
                traceback.print_exc()

    return resp_dict, stim_dict_speech, stim_dict_music, freqs


# Main STRF function
def strf_stacked_speechmusic_refit(subject, data_dir, test_set,
                                   fs=100.0, delay_min=0.0, delay_max=1.0):
    """
    Refit stacked speech+music STRF with updated train/test split.
    Saves to strfs_stacked_speechmusic_refit/ to avoid overwriting original.

    Parameters
    ----------
    subject  : str   — subject ID
    data_dir : str   — root data directory
    test_set : list  — list of wav filenames to use as test set
    fs       : float — sampling rate in Hz (default 100.0)
    delay_min, delay_max : float — delay range in seconds
    """
    output_dir = f'{data_dir}/sub-{subject}/strfs_stacked_speechmusic'
    os.makedirs(output_dir, exist_ok=True)

    resp_dict, stim_dict_speech, stim_dict_music, freqs = loadEEGh5_stacked(
        subject, data_dir, resp_mean=True)

    all_stimuli  = [k for k in stim_dict_speech.keys() if len(resp_dict[k]) > 0]
    training_set = [s for s in all_stimuli if s not in test_set]

    print(f"Training set ({len(training_set)}): {training_set}")
    print(f"Test set ({len(test_set)}): {test_set}")

    # Build stimulus and response matrices
    tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
    vResp = np.hstack([resp_dict[r][0] for r in test_set]).T

    tStim_speech = np.atleast_2d(np.vstack([np.vstack(stim_dict_speech[r]).T for r in training_set]))
    vStim_speech = np.atleast_2d(np.vstack([np.vstack(stim_dict_speech[r]).T for r in test_set]))
    tStim_music  = np.atleast_2d(np.vstack([np.vstack(stim_dict_music[r]).T  for r in training_set]))
    vStim_music  = np.atleast_2d(np.vstack([np.vstack(stim_dict_music[r]).T  for r in test_set]))

    for arr in [tStim_speech, vStim_speech, tStim_music, vStim_music]:
        arr[np.isnan(arr)] = 0

    # Stack speech + music (160 features total)
    tStim_temp = np.hstack([tStim_speech, tStim_music])
    vStim_temp = np.hstack([vStim_speech, vStim_music])

    delays = np.arange(np.floor(delay_min * fs), np.ceil(delay_max * fs), dtype=int)
    tStim  = make_delayed(tStim_temp, delays)
    vStim  = make_delayed(vStim_temp, delays)

    print(f"tStim: {tStim.shape}, tResp: {tResp.shape}")
    print(f"vStim: {vStim.shape}, vResp: {vResp.shape}")

    # Ridge regression
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

    # Save
    channel_names_file = f'{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt'
    channel_names = np.loadtxt(channel_names_file, usecols=(0), dtype=str)

    # encode test/train split in filename so it's clear what was used
    strf_file = f'{output_dir}/STRF_by_stacked_speechmusic_spec.hf5'
    print(f"Saving to {strf_file}")

    train_inds = np.array([i for i, s in enumerate(all_stimuli) if s in training_set])
    val_inds   = np.array([i for i, s in enumerate(all_stimuli) if s in test_set])

    with h5py.File(strf_file, 'w') as f:
        f.create_dataset('/channel_names', data=np.array(channel_names, dtype=h5py.string_dtype()))
        f.create_dataset('/wts',           data=wt)
        f.create_dataset('/corrs',         data=corrs)
        f.create_dataset('/valphas',       data=valphas)
        f.create_dataset('/allRcorrs',     data=allRcorrs)
        f.create_dataset('/train_inds',    data=train_inds)
        f.create_dataset('/val_inds',      data=val_inds)
        f.create_dataset('/delays',        data=delays)
        f.create_dataset('/freqs',         data=freqs)
        # save test/train filenames for provenance
        f.create_dataset('/test_set',      data=np.array(test_set,     dtype=h5py.string_dtype()))
        f.create_dataset('/training_set',  data=np.array(training_set, dtype=h5py.string_dtype()))

    print("Done.")
    return wt, corrs, valphas, allRcorrs


# Run
if __name__ == "__main__":
    user = 'rajviagravat'

    subjects = {
        'TCH43': ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav'],
        'TCH45': ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']
    }

    for subject, test_set in subjects.items():
        data_dir = f'/Users/{user}/Library/CloudStorage/Box-Box/{"TCH_ECoG" if subject.startswith("TCH") else "ECoG_Backup"}'
        print(f"\nProcessing {subject}")
        wt, corrs, valphas, allRcorrs = strf_stacked_speechmusic_refit(
            subject, data_dir, test_set,
            fs=100.0, delay_min=0.0, delay_max=1.0
        )