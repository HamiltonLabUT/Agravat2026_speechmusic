import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import pandas as pd
from matplotlib import cm

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

subject_list = ['S0005','S0006','S0009','S0011','S0012','S0013','S0016','S0018',
                'S0021','S0022','S0024','S0025','S0027','S0030','S0031','S0033','S0036',
                'S0038','S0039','S0040','TCH2','TCH5','TCH06','TCH11','TCH13',
                'TCH14','TCH15','TCH16','TCH18','TCH19','TCH20','TCH22','TCH28','TCH29',
                'TCH30','TCH37','TCH38','TCH39','TCH42','TCH49',
                'TCH50','TCH51','TCH52','TCH53','TCH56',
                'TCH58','TCH61','TCH62','TCH64','TCH65','TCH66','TCH69']

SAVE_DIR = '/Users/rajviagravat/Library/CloudStorage/Box-Box/Figures/speechmusic/DNN_analysis/webviewer_RFs/temporal_elecs_only'

CSV_PATH = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis/DNN_analysis_allmodels_02_26.csv'

TEMPORAL_REGIONS = ['STG', 'STS', 'MTG', 'HG', 'PT', 'PP']

def get_data_dir(subject):
    if subject.startswith('TCH'):
        return '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_ECoG'
    elif subject.startswith('S'):
        return '/Users/rajviagravat/Library/CloudStorage/Box-Box/ECoG_backup'
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_strf(hf5_path, wts_key, corrs_key, nfeats):
    with h5py.File(hf5_path, 'r') as hf:
        wts = hf[wts_key][:]
        corrs = hf[corrs_key][:]
    wts2 = wts.reshape(wts.shape[0] // nfeats, nfeats, wts.shape[1])
    return wts2, corrs


def draw_strf(ax, strf, title, corr, delay_min, delay_max):

    n_freqs = strf.shape[0]
    strf_flipped = np.fliplr(strf)
    local_max = np.abs(strf_flipped).max()

    im = ax.imshow(strf_flipped,
                   cmap=cm.RdBu_r,
                   aspect='auto',
                   interpolation='nearest',
                   vmin=-local_max,
                   vmax=local_max,
                   origin='lower',
                   extent=[delay_min, delay_max, -0.5, n_freqs - 0.5])

    ax.set_xlim(delay_min, delay_max)
    mid_time = (delay_min + delay_max) / 2
    ax.set_xticks([delay_min, mid_time, delay_max])
    ax.set_xticklabels([f'{delay_min:.0f}', f'{mid_time:.1f}', f'{delay_max:.0f}'])
    ax.set_xlabel('Time (s)')

    ax.set_yticks([0, 40, 79])
    ax.set_yticklabels(['0.5', '2', '8'])

    ax.set_title(f'{title}\nr = {corr:.3g}', fontsize=9)

    return im


def make_figure(strfs, corrs_vals, titles,
                subject, chname,
                shared_max,
                delay_min, delay_max):

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    fig.subplots_adjust(left=0.19, right=0.95, top=0.78, bottom=0.2, wspace=0.4)

    for ax, strf, title, corr in zip(axes, strfs, titles, corrs_vals):
        draw_strf(ax, strf, title, corr, delay_min, delay_max)
        ax.set_ylim(-0.5, 79.5)

    # Shared colorbar using a ScalarMappable tied to shared_max
    norm = mcolors.Normalize(vmin=-shared_max, vmax=shared_max)
    sm = cm.ScalarMappable(cmap=cm.RdBu_r, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.06, 0.2, 0.02, 0.55])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([-shared_max, shared_max])
    cbar.set_ticklabels(['-max', 'max'])
    cbar.ax.set_ylabel('STRF weights', fontsize=8, labelpad=-20)

    axes[-1].yaxis.set_label_position("right")
    axes[-1].set_ylabel('Freq (kHz)')

    fig.suptitle(f'{subject}  {chname}', fontsize=11, fontweight='bold')

    return fig


def save_strfs(subject_list, fs=100.0, delay_min=-1.0, delay_max=0.0):

    ensure_dir(SAVE_DIR)

    if not os.path.exists(CSV_PATH):
        print("Anatomy CSV not found. Exiting.")
        return

    anat_df = pd.read_csv(CSV_PATH)
    anat_df = anat_df[anat_df['short_anat'].isin(TEMPORAL_REGIONS)]

    for subject in subject_list:

        print(f"\nProcessing {subject}...")

        data_dir = get_data_dir(subject)
        if data_dir is None:
            print("  Skipping: unknown prefix.")
            continue

        subj_anat = anat_df[anat_df['subj_id'] == subject]
        if len(subj_anat) == 0:
            print("  Skipping: no temporal electrodes in CSV.")
            continue

        temporal_channels = set(subj_anat['channelnames'].values)

        chnames_path = f'{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt'
        if not os.path.exists(chnames_path):
            print("  Skipping: channel names file missing.")
            continue

        chnames = np.loadtxt(chnames_path, usecols=(0), dtype='str')

        base_og = f'{data_dir}/sub-{subject}/strfs_og'
        base_sm = f'{data_dir}/sub-{subject}/strfs_speechmusic'

        path_mixed = find_file([
            f'{base_og}/STRF_by_spec_MT_shifted.hf5',
            f'{base_og}/STRF_by_spec_MT.hf5',
        ])
        path_speech = find_file([
            f'{base_sm}/STRF_by_spec_MT_speech_shifted.hf5',
            f'{base_sm}/STRF_by_spec_MT_speech.hf5',
        ])
        path_music = find_file([
            f'{base_sm}/STRF_by_spec_MT_music_shifted.hf5',
            f'{base_sm}/STRF_by_spec_MT_music.hf5',
        ])

        if None in [path_mixed, path_speech, path_music]:
            print("  Skipping: missing STRF files.")
            continue

        try:
            wts_mixed,  corrs_mixed  = load_strf(path_mixed,  '/wts',    '/corrs',    nfeats=80)
            wts_speech, corrs_speech = load_strf(path_speech, '/wts_mt', '/corrs_mt', nfeats=80)
            wts_music,  corrs_music  = load_strf(path_music,  '/wts_mt', '/corrs_mt', nfeats=80)
        except Exception as e:
            print(f"  Skipping: error loading STRFs ({e})")
            continue

        if not (wts_mixed.shape[2] == wts_speech.shape[2] == wts_music.shape[2]):
            print("  Skipping: channel count mismatch across STRFs.")
            continue

        n_chs = wts_mixed.shape[2]

        overlapping = temporal_channels.intersection(set(chnames))
        if len(overlapping) == 0:
            print("  Skipping: no matching channel names.")
            continue

        print(f"  Plotting {len(overlapping)} temporal electrodes...")

        for m in range(n_chs):

            chname = chnames[m]
            if chname not in overlapping:
                continue

            strf_mixed  = wts_mixed[:, :, m].T
            strf_speech = wts_speech[:, :, m].T
            strf_music  = wts_music[:, :, m].T

            shared_max = max(np.abs(strf_mixed).max(),
                             np.abs(strf_speech).max(),
                             np.abs(strf_music).max())

            fig = make_figure(
                strfs=[strf_mixed, strf_speech, strf_music],
                corrs_vals=[corrs_mixed[m],
                            corrs_speech[m],
                            corrs_music[m]],
                titles=['Mixed', 'Speech\nseparated', 'Music\nseparated'],
                subject=subject,
                chname=chname,
                shared_max=shared_max,
                delay_min=delay_min,
                delay_max=delay_max
            )

            fig.savefig(os.path.join(SAVE_DIR,
                        f'{subject}_{chname}_3panel.png'),
                        dpi=150, bbox_inches='tight')

            plt.close(fig)

        print(f"  Done: {subject}")


save_strfs(subject_list, fs=100.0, delay_min=-1.0, delay_max=0.0)