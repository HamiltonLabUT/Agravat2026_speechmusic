import os
import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

# Set Times New Roman as the default font
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18 

# Helper function for rounding
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# List of subjects
subject_list = [
'TCH43','TCH45'
]

# Function to calculate plot grid size (rows, cols) based on len(picks)
def calculate_grid_size(num_channels):
    cols = math.ceil(math.sqrt(num_channels))  # Number of columns
    rows = math.ceil(num_channels / cols)  # Number of rows
    return rows, cols

# Direct implementation of the plot_strf function to avoid dependency issues
def plot_strf_direct(data_dir, subject, save_dir, strf_type, subplot_r, subplot_c,
              stimulus_class, fs=100.0, delay_min=0.0, delay_max=1.0, plot_all_chs=True):
    """
    Direct implementation of plot_strf to avoid module import issues
    """
    # Check whether the output path to save strfs exists or not
    output_dir = f'/{data_dir}/sub-{subject}/strfs_speechmusic/plot_figs'
    isExist = os.path.exists(output_dir)

    if not isExist:
        os.makedirs(output_dir)
        print("The new directory is created!")

    if strf_type == 'spectrogram':
        strf_filename = 'spec'
        feat_labels = np.arange(80).tolist()

    with h5py.File('%s/sub-%s/strfs_speechmusic/STRF_by_%s_MT_%s.hf5'%(data_dir, subject, strf_filename, stimulus_class),'r') as hf:
        wts = hf['/wts_mt'][:]
        corrs = hf['/corrs_mt'][:]

    nfeats = len(feat_labels)
    chnames = np.loadtxt(f'/{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt', usecols=(0), dtype='str')
    print(f"Number of channels: {len(chnames)}")

    wts2 = wts.reshape(int(wts.shape[0]/nfeats),nfeats,wts.shape[1]) #reshape weights since they are not from fitting STRFs
    print(f"Weights shape: {wts2.shape}")

    delays = np.arange(np.floor((delay_min)*fs), np.ceil((delay_max)*fs), dtype=int) #create array to pass time delays in
    print("Delays:", delays)

    #create subplot:
    fig = plt.figure(figsize=(45,50))
    if plot_all_chs:
        for m in range((wts2.shape[2])):
            plt.subplot(subplot_r, subplot_c, m+1)
            strf = wts2[:,:,m].T

            smax = np.abs(strf).max()
            t = np.linspace(delay_min, delay_max, len(delays))
            plt.imshow(strf, cmap=cm.RdBu_r, aspect='auto', interpolation='nearest', vmin=-smax, vmax=smax)
            plt.gca().invert_xaxis()
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
            plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
            plt.gca().set_xticklabels([t[0], round_up(t[int((len(delays)-1)/2)],2), t[len(delays)-1]], fontsize = 18)
            if strf_type == 'spectrogram':
                plt.gca().invert_yaxis()
                plt.gca().set_yticks([0, 40, 80])
                plt.gca().set_yticklabels([0.5, 2, 8], fontsize=18)
            else:
                plt.gca().set_yticks(np.arange(strf.shape[0]))
                plt.gca().set_yticklabels(feat_labels, fontsize=18)

            plt.title('%s r=%.3g'%(chnames[m], corrs[m]), fontsize=20, pad=20)
            plt.colorbar(location='left')
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        plt.tight_layout()
        plt.savefig('%s/sub-%s/strfs_speechmusic/plot_figs/%s_%s_wts_STRF_subplots.pdf' %(save_dir, subject, stimulus_class, strf_type))

# Function to plot STRFs for multiple subjects and both stimulus types
def plot_strfs_for_multiple_subjects(subject_list, fs=100.0, delay_min=0.0, delay_max=1.0):
    completed_subjects = []  # List to store completed subjects
    
    for subject in subject_list:  # Iterate through the subject list
        # Determine the data and save directories based on subject ID prefix
        if subject.startswith('TCH'):
            data_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_ECoG/'
            save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/TCH_ECoG/'
        elif subject.startswith('S'):
            data_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/ECoG_backup/'
            save_dir = '/Users/rajviagravat/Library/CloudStorage/Box-Box/ECoG_backup/'
        else:
            print(f"Unknown subject prefix for {subject}. Skipping...")
            continue  # Skip the subject if the prefix is not recognized
        
        strf_type = 'spectrogram'  # 'spectrogram' or 'binned'
        
        # Load the picks for the subject (electrodes)
        picks_path = f'{data_dir}/sub-{subject}/{subject}_channelnames_speech_music.txt'
        if not os.path.exists(picks_path):
            print(f"File not found for subject {subject}: {picks_path}")
            continue  # Skip this subject if file is missing
            
        picks = np.loadtxt(picks_path, usecols=(0), dtype='str')
        len_picks = len(picks)
        rows, cols = calculate_grid_size(len_picks)
        print(f"Processing subject: {subject}, len(picks)={len_picks}, grid={rows}x{cols}")
        
        # Loop through both speech and music stimuli
        for stimulus_class in ['speech', 'music']:
            print(f"Plotting for {stimulus_class} stimulus for subject {subject}")
            
            try:
                # Call our direct implementation instead of relying on the module
                plot_strf_direct(
                    data_dir=data_dir,
                    subject=subject,
                    save_dir=save_dir,
                    strf_type=strf_type,
                    subplot_r=rows,
                    subplot_c=cols,
                    stimulus_class=stimulus_class,
                    fs=fs,
                    delay_min=delay_min,
                    delay_max=delay_max,
                    plot_all_chs=True
                )
                print(f"Successfully plotted {stimulus_class} for {subject}")
            except Exception as e:
                print(f"Error plotting {stimulus_class} for {subject}: {str(e)}")
                # Print more information about the error
                import traceback
                traceback.print_exc()
        
        # Mark this subject as completed
        completed_subjects.append(subject)
        print(f"Completed STRF plot for subject: {subject}")
        print(f"Subjects completed so far: {', '.join(completed_subjects)}")

# Example usage
# Plot STRFs for all subjects in the list and for both speech and music stimuli
plot_strfs_for_multiple_subjects(subject_list, fs=100.0, delay_min=0.0, delay_max=1.0)