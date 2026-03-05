import os
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18 

# Helper function for rounding
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# List of subjects
subject_list = [
'TCH64'
]

# Function to calculate plot grid size (rows, cols) based on len(picks)
def calculate_grid_size(num_channels):
    cols = math.ceil(math.sqrt(num_channels))  # Number of columns
    rows = math.ceil(num_channels / cols)  # Number of rows
    return rows, cols

# plot_strf function for stacked analysis
def plot_strf_stacked_direct(data_dir, subject, save_dir, subplot_r, subplot_c,
                            fs=100.0, delay_min=0.0, delay_max=1.0, plot_all_chs=True):
    """
    Plot stacked speech+music STRFs (160 features)
    """
    # Check whether the output path to save strfs exists or not
    output_dir = f'/{data_dir}/sub-{subject}/strfs_stacked_speechmusic/plot_figs'
    isExist = os.path.exists(output_dir)

    if not isExist:
        os.makedirs(output_dir)
        print("The new directory is created!")

    # Load the stacked STRF file
    with h5py.File('%s/sub-%s/strfs_stacked_speechmusic/STRF_by_stacked_speechmusic_spec.hf5'%(data_dir, subject),'r') as hf:
        wts = hf['/wts'][:]
        corrs = hf['/corrs'][:]

    nfeats = 160  # 80 speech + 80 music features
    chnames = np.loadtxt(f'/{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt', usecols=(0), dtype='str')
    print(f"Number of channels: {len(chnames)}")

    wts2 = wts.reshape(int(wts.shape[0]/nfeats), nfeats, wts.shape[1])
    print(f"Weights shape: {wts2.shape}")

    delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=int)
    print("Delays:", delays)

    # Create subplot:
    fig = plt.figure(figsize=(45,50))
    if plot_all_chs:
        for m in range((wts2.shape[2])):
            plt.subplot(subplot_r, subplot_c, m+1)
            strf = wts2[:,:,m].T

            smax = np.abs(strf).max()
            t = np.linspace(delay_min, delay_max, len(delays))
            plt.imshow(strf, cmap=cm.RdBu_r, aspect='auto', interpolation='nearest', 
                      vmin=-smax, vmax=smax, origin='lower')
            plt.gca().invert_xaxis()
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
            plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
            plt.gca().set_xticklabels([t[0], round_up(t[int((len(delays)-1)/2)],2), t[len(delays)-1]], fontsize = 18)
            
            # Set y-axis ticks for stacked features (speech: 0-79, music: 80-159)
            plt.gca().set_yticks([0, 40, 79, 80, 120, 159])
            plt.gca().set_yticklabels(['Speech\n0.5kHz', '2kHz', '8kHz', 'Music\n0.5kHz', '2kHz', '8kHz'], fontsize=16)
            
            # Add horizontal line to separate speech and music
            plt.axhline(y=79.5, color='white', linestyle='--', linewidth=2, alpha=0.8)

            plt.title('%s r=%.3g'%(chnames[m], corrs[m]), fontsize=20, pad=20)
            plt.colorbar(location='left')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        plt.tight_layout()
        plt.savefig('%s/sub-%s/strfs_stacked_speechmusic/plot_figs/stacked_speechmusic_wts_STRF_subplots.pdf' %(save_dir, subject))

# Function to plot stacked STRFs for multiple subjects
def plot_stacked_strfs_for_multiple_subjects(subject_list, fs=100.0, delay_min=0.0, delay_max=1.0):
    completed_subjects = []  # List to store completed subjects
    
    for subject in subject_list:  # Iterate through the subject list
        # Determine the data and save directories based on subject ID prefix
        if subject.startswith('TCH'):
            data_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_ECoG'
            save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_ECoG'
        elif subject.startswith('S'):
            data_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/ECoG_backup/'
            save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/ECoG_backup/'
        else:
            print(f"Unknown subject prefix for {subject}. Skipping...")
            continue  # Skip the subject if the prefix is not recognized
        
        # Load the picks for the subject (electrodes)
        picks_path = f'{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt'
        if not os.path.exists(picks_path):
            print(f"File not found for subject {subject}: {picks_path}")
            continue  # Skip this subject if file is missing
            
        picks = np.loadtxt(picks_path, usecols=(0), dtype='str')
        len_picks = len(picks)
        rows, cols = calculate_grid_size(len_picks)
        print(f"Processing subject: {subject}, len(picks)={len_picks}, grid={rows}x{cols}")
        
        print(f"Plotting stacked STRF for subject {subject}")
        
        try:
            plot_strf_stacked_direct(
                data_dir=data_dir,
                subject=subject,
                save_dir=save_dir,
                subplot_r=rows,
                subplot_c=cols,
                fs=fs,
                delay_min=delay_min,
                delay_max=delay_max,
                plot_all_chs=True
            )
            print(f"Successfully plotted stacked STRF for {subject}")
        except Exception as e:
            print(f"Error plotting stacked STRF for {subject}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Mark this subject as completed
        completed_subjects.append(subject)
        print(f"Completed stacked STRF plot for subject: {subject}")
        print(f"Subjects completed so far: {', '.join(completed_subjects)}")

# Plot stacked STRFs for all subjects in the list
plot_stacked_strfs_for_multiple_subjects(subject_list, fs=100.0, delay_min=0.0, delay_max=1.0)