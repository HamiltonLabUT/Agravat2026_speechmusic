import os
import sys
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib
# matplotlib.use('Qt5Agg')  # Commented out - not needed for CSV generation
import mplcursors
from scipy.io import loadmat
import h5py
from datetime import datetime
ridge_dir = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/ridge'
utils_dir = '/Users/rajviagravat/Documents/Hamilton_Lab/Code/speechMusic/analysis/ridge'
sys.path.append(utils_dir)
sys.path.append(ridge_dir)
import ridge_
from ridge_ import ridge
import utils
from utils import make_delayed
from adjustText import adjust_text

today_datetime = datetime.today().strftime('%m_%d')
zs = lambda x: (x-x[np.isnan(x)==False].mean(0))/x[np.isnan(x)==False].std(0)

def dnn_analysis(subj_id, ecog_speechmusic_path, ecog_movietrailers_path, strf_speech_path, 
                 strf_music_path, strf_speechmusic_path, strf_stacked_path=None,
                 strf_attend_speech_path=None, strf_attend_music_path=None):
    """
    Load pre-computed correlations for a subject from all model h5 files.
    Returns correlations for speech-only, music-only, speech-music, stacked, 
    attend-speech, and attend-music conditions.
    
    IMPORTANT: Validates that channel names in h5 files match expected channel names
    to ensure correlations are aligned correctly.
    """
    
    # Initialize lists to store correlations
    speech_only_corrs_DNN = []
    music_only_corrs_DNN = []
    speech_music_corrs_DNN = []
    stacked_corrs_DNN = []
    attend_speech_corrs_DNN = []
    attend_music_corrs_DNN = []

    def validate_channels(h5_file, expected_channels, model_name):
        """
        Validate that channel names in h5 file match expected channel names.
        Returns True if valid, raises error if mismatch.
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                if '/channel_names' in f:
                    h5_channels = [ch.decode('utf-8') if isinstance(ch, bytes) else ch 
                                   for ch in f['channel_names'][:]]
                    
                    if len(h5_channels) != len(expected_channels):
                        print(f"WARNING: {model_name} has {len(h5_channels)} channels, "
                              f"but expected {len(expected_channels)} channels!")
                        print(f"  H5 channels: {h5_channels[:5]}... (showing first 5)")
                        print(f"  Expected channels: {expected_channels[:5]}... (showing first 5)")
                        return False
                    
                    mismatches = []
                    for i, (h5_ch, exp_ch) in enumerate(zip(h5_channels, expected_channels)):
                        if h5_ch != exp_ch:
                            mismatches.append((i, h5_ch, exp_ch))
                    
                    if mismatches:
                        print(f"WARNING: {model_name} has channel name mismatches!")
                        for idx, h5_ch, exp_ch in mismatches[:5]:  # Show first 5
                            print(f"  Channel {idx}: H5='{h5_ch}' vs Expected='{exp_ch}'")
                        return False
                    
                    print(f"✓ {model_name}: Channel names validated ({len(h5_channels)} channels)")
                    return True
                else:
                    print(f"WARNING: {model_name} h5 file does not contain /channel_names dataset")
                    return False
        except Exception as e:
            print(f"ERROR validating {model_name}: {e}")
            return False

    # Load expected channel names from text file
    base_dir = '/Users/rajviagravat'
    channel_names_paths = [
        f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{subj_id}_channelnames_speech_music.txt',
        f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{subj_id}_channelnames_speech_music.txt'
    ]
    
    expected_channels = None
    for path in channel_names_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                expected_channels = [line.strip() for line in f.readlines()]
            break
    
    if expected_channels is None:
        print(f"ERROR: Could not find channel names file for {subj_id}")
        return ([], [], [], [], [], [])
    
    print(f"\n{'='*60}")
    print(f"Loading correlations for {subj_id}")
    print(f"Expected number of channels: {len(expected_channels)}")
    print(f"{'='*60}\n")

    # Load speech-only correlations
    print(f"Loading speech-only correlations for {subj_id}")
    validate_channels(strf_speech_path, expected_channels, "Speech-only")
    with h5py.File(strf_speech_path, 'r') as f:
        speech_only_corrs_DNN = f['corrs_mt'][:].tolist()
        print(f"  Loaded {len(speech_only_corrs_DNN)} speech-only correlations")

    # Load music-only correlations
    print(f"\nLoading music-only correlations for {subj_id}")
    validate_channels(strf_music_path, expected_channels, "Music-only")
    with h5py.File(strf_music_path, 'r') as f:
        music_only_corrs_DNN = f['corrs_mt'][:].tolist()
        print(f"  Loaded {len(music_only_corrs_DNN)} music-only correlations")

    # Load speech-music correlations
    print(f"\nLoading speech-music correlations for {subj_id}")
    validate_channels(strf_speechmusic_path, expected_channels, "Speech-music")
    with h5py.File(strf_speechmusic_path, 'r') as f:
        speech_music_corrs_DNN = f['corrs'][:].tolist()
        print(f"  Loaded {len(speech_music_corrs_DNN)} speech-music correlations")

    # Load stacked correlations if available
    if strf_stacked_path and os.path.exists(strf_stacked_path):
        try:
            print(f"\nLoading stacked correlations for {subj_id}")
            validate_channels(strf_stacked_path, expected_channels, "Stacked")
            with h5py.File(strf_stacked_path, 'r') as f:
                stacked_corrs_DNN = f['corrs'][:].tolist()
                print(f"  Loaded {len(stacked_corrs_DNN)} stacked correlations")
        except Exception as e:
            print(f"Error loading stacked correlations for {subj_id}: {e}")
            stacked_corrs_DNN = []
    else:
        print(f"\nStacked STRF file not found for {subj_id}, skipping stacked correlations")
        stacked_corrs_DNN = []

    # Final validation: Check that all loaded correlations have the same length
    corr_lengths = {
        'speech_only': len(speech_only_corrs_DNN),
        'music_only': len(music_only_corrs_DNN),
        'speech_music': len(speech_music_corrs_DNN),
        'stacked': len(stacked_corrs_DNN) if stacked_corrs_DNN else None,
        # 'attend_speech': len(attend_speech_corrs_DNN) if attend_speech_corrs_DNN else None,
        # 'attend_music': len(attend_music_corrs_DNN) if attend_music_corrs_DNN else None
    }
    
    print(f"\n{'='*60}")
    print(f"Summary for {subj_id}:")
    for model, length in corr_lengths.items():
        if length is not None:
            status = "✓" if length == len(expected_channels) else "✗ MISMATCH"
            print(f"  {model}: {length} correlations {status}")
    print(f"{'='*60}\n")

    return (speech_only_corrs_DNN, music_only_corrs_DNN, speech_music_corrs_DNN, 
            stacked_corrs_DNN, attend_speech_corrs_DNN, attend_music_corrs_DNN)

def get_rois():
    """Get ROI mappings"""
    all_rois = [
        # new wmparc rois
        {'anat_label': 'wm-lh-transversetemporal', 'short_anat': 'HG'},
        {'anat_label': 'wm-rh-transversetemporal-G_T_transv', 'short_anat': 'HG'},
        {'anat_label': 'wm-lh-superiortemporal', 'short_anat': 'STG'},
        {'anat_label': 'wm-rh-superiortemporal', 'short_anat': 'STG'},
        {'anat_label': 'wm-rh-bankssts', 'short_anat': 'STS'},
        {'anat_label': 'wm-lh-bankssts', 'short_anat': 'STS'},
        {'anat_label': 'wm-lh-middletemporal', 'short_anat': 'MTG'},
        {'anat_label': 'wm-rh-middletemporal', 'short_anat': 'MTG'},
        {'anat_label': 'wm-lh-insula', 'short_anat': 'insula'},
        {'anat_label': 'wm-rh-insula', 'short_anat': 'insula'},
        {'anat_label': 'ctx_lh_G_insular_short', 'short_anat': 'insula'},
        {'anat_label': 'ctx_rh_G_insular_short', 'short_anat': 'insula'},
        {'anat_label': 'ctx_lh_S_circular_insula_ant', 'short_anat': 'insula'},
        {'anat_label': 'ctx_rh_S_circular_insula_ant', 'short_anat': 'insula'},
        {'anat_label': 'Left-Thalamus', 'short_anat': 'other'},
        {'anat_label': 'Right-Thalamus', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_G_front_middle', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_G_front_middle', 'short_anat': 'other'},
        {'anat_label': 'Left-Cerebral-White-Matter', 'short_anat': 'other'},
        {'anat_label': 'Right-Cerebral-White-Matter', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_G_precentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_G_precentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_S_precentral-sup-part', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_S_precentral-sup-part', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_S_precentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_S_precentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_G_postcentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_G_postcentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_S_postcentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_S_postcentral', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_G_temp_sup-Lateral', 'short_anat': 'STG'},
        {'anat_label': 'ctx_lh_G_temporal_middle', 'short_anat': 'MTG'},
        {'anat_label': 'ctx_lh_G_temp_sup-Plan_tempo', 'short_anat': 'PT'},
        {'anat_label': 'ctx_lh_G_temp_sup-Plan_polar', 'short_anat': 'PP'},
        {'anat_label': 'ctx_lh_S_temporal_sup', 'short_anat': 'STS'},
        {'anat_label': 'ctx_lh_G_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'ctx_lh_G_temp_sup-G_T_transv', 'short_anat': 'HG'},
        {'anat_label': 'ctx_lh_S_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'superiortemporal', 'short_anat': 'STG'},
        {'anat_label': 'ctx_lh_G_pariet_inf-Supramar', 'short_anat': 'other'},
        {'anat_label': 'ctx_lh_Lat_Fis-post', 'short_anat': 'STG'},
        {'anat_label': 'ctx_lh_S_circular_insula_inf', 'short_anat': 'insula'},
        {'anat_label': 'ctx_rh_G_temp_sup-Lateral', 'short_anat': 'STG'},
        {'anat_label': 'ctx_rh_G_temporal_middle', 'short_anat': 'MTG'},
        {'anat_label': 'ctx_rh_G_temp_sup-Plan_tempo', 'short_anat': 'PT'},
        {'anat_label': 'ctx_rh_G_temp_sup-Plan_polar', 'short_anat': 'PP'},
        {'anat_label': 'ctx_rh_S_temporal_sup', 'short_anat': 'STS'},
        {'anat_label': 'ctx_rh_G_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'ctx_rh_G_temp_sup-G_T_transv', 'short_anat': 'HG'},
        {'anat_label': 'ctx_rh_S_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'ctx_rh_G_pariet_inf-Supramar', 'short_anat': 'other'},
        {'anat_label': 'ctx_rh_Lat_Fis-post', 'short_anat': 'STG'},
        {'anat_label': 'ctx_rh_S_circular_insula_inf', 'short_anat': 'insula'},
        {'anat_label': 'ctx_lh_S_circular_insula_sup', 'short_anat': 'insula'},
        {'anat_label': 'ctx_rh_S_circular_insula_sup', 'short_anat': 'insula'},
        {'anat_label': 'ctx_lh_G_Ins_lg_and_S_cent_ins', 'short_anat': 'insula'},
        {'anat_label': 'ctx_rh_G_Ins_lg_and_S_cent_ins', 'short_anat': 'insula'}
    ]
    
    all_rois_no_hem = [
        {'anat_label': 'Left-Thalamus', 'short_anat': 'other'},
        {'anat_label': 'Right-Thalamus', 'short_anat': 'other'},
        {'anat_label': 'G_front_middle', 'short_anat': 'other'},
        {'anat_label': 'Left-Cerebral-White-Matter', 'short_anat': 'other'},
        {'anat_label': 'Right-Cerebral-White-Matter', 'short_anat': 'other'},
        {'anat_label': 'G_precentral', 'short_anat': 'other'},
        {'anat_label': 'S_precentral-sup-part', 'short_anat': 'other'},
        {'anat_label': 'S_precentral', 'short_anat': 'other'},
        {'anat_label': 'G_postcentral', 'short_anat': 'other'},
        {'anat_label': 'S_postcentral', 'short_anat': 'other'},
        {'anat_label': 'G_temp_sup-Lateral', 'short_anat': 'STG'},
        {'anat_label': 'G_temporal_middle', 'short_anat': 'MTG'},
        {'anat_label': 'G_temp_sup-Plan_tempo', 'short_anat': 'PT'},
        {'anat_label': 'G_temp_sup-Plan_polar', 'short_anat': 'PP'},
        {'anat_label': 'S_temporal_sup', 'short_anat': 'STS'},
        {'anat_label': 'G_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'G_temp_sup-G_T_transv', 'short_anat': 'HG'},
        {'anat_label': 'S_temporal_transverse', 'short_anat': 'HG'},
        {'anat_label': 'superiortemporal', 'short_anat': 'STG'},
        {'anat_label': 'G_pariet_inf-Supramar', 'short_anat': 'other'},
        {'anat_label': 'Lat_Fis-post', 'short_anat': 'STG'},
        {'anat_label': 'S_circular_insula_inf', 'short_anat': 'insula'},
        {'anat_label': 'S_circular_insula_sup', 'short_anat': 'insula'},
        {'anat_label': 'G_Ins_lg_and_S_cent_ins', 'short_anat': 'insula'}
    ]

    speech_rois = all_rois  # Using all_rois for speech ROIs
    speech_rois_no_ctx_hem = all_rois_no_hem  # Using all_rois_no_hem for speech ROIs without hemisphere
    
    return all_rois, all_rois_no_hem, speech_rois, speech_rois_no_ctx_hem

def get_roi_mapping():
    """Get ROI full names mapping"""
    roi_full_names = {
        'HG': 'Heschl\'s Gyrus',
        'STG': 'Superior Temporal Gyrus',
        'MTG': 'Middle Temporal Gyrus',
        'STS': 'Superior Temporal Sulcus',
        'PP': 'Planum Polare',
        'PT': 'Planum Temporale',
        'insula': 'Insula'
    }
    return roi_full_names

def get_dev_stage(age):
    """Map age to developmental stage"""
    if isinstance(age, pd.Series):  # Ensure age is not a Series
        age = age.iloc[0]  # Extract the first value if it's a Series
    if 4 <= age <= 5:
        return "Early childhood"
    elif 6 <= age <= 11:
        return "Middle childhood"
    elif 12 <= age <= 17:
        return "Early adolescence"
    else:
        return "Late adolescence"
        
def create_dnn_dataframe(subj_id, anatomy_data, participant_info, dnn_correlations, site):
    """
    Create a DataFrame for DNN analysis by matching channel names from 
    channelnames_speech_music.txt with tdt_elecs_all_combined.mat and appending 
    the corresponding data.
    """
    # Get subject info
    subj_info = participant_info[participant_info['subj_id'] == subj_id].iloc[0]
    age = subj_info['age']
    sex = subj_info['sex']
    dev_stage = get_dev_stage(age)

    # Get ROI mapping
    all_rois, _, _, _ = get_rois()

    # Extract anatomical labels and channel names from anatomy_data (tdt_elecs_all_combined.mat)
    anatomy_labels = anatomy_data['anatomy'][:, 3]
    channelnames = anatomy_data['anatomy'][:, 0]
    elecmatrix = anatomy_data['elecmatrix']

    # Ensure channelnames is a flat array of strings
    if isinstance(channelnames, np.ndarray):
        if channelnames.dtype == object:
            channelnames = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in channelnames])
        else:
            channelnames = channelnames.astype(str)

    # Define possible paths for channelnames_speech_music.txt
    base_dir = '/Users/rajviagravat'
    speech_music_paths = [
        f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{subj_id}_channelnames_speech_music.txt',
        f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{subj_id}_channelnames_speech_music.txt'
    ]

    # Load channel names from channelnames_speech_music.txt (check both paths)
    speech_music_ch_names = None
    for path in speech_music_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                speech_music_ch_names = [line.strip() for line in f.readlines()]
            break

    if speech_music_ch_names is None:
        print(f"Speech-music channel names file not found for subject {subj_id}.")
        return pd.DataFrame()

    # Create a mapping from tdt_elecs_all_combined.mat channel names to their indices
    ch_name_to_index = {ch: idx for idx, ch in enumerate(channelnames)}

    # Create dataframe rows
    rows = []

    # Print lengths for debugging
    print(f"Length of speech_only_corrs_DNN: {len(dnn_correlations['speech_only_corrs_DNN'])}")
    print(f"Length of music_only_corrs_DNN: {len(dnn_correlations['music_only_corrs_DNN'])}")
    print(f"Length of speech_music_corrs_DNN: {len(dnn_correlations['speech_music_corrs_DNN'])}")
    print(f"Length of stacked_corrs_DNN: {len(dnn_correlations['stacked_corrs_DNN'])}")
    # print(f"Length of attend_speech_corrs_DNN: {len(dnn_correlations['attend_speech_corrs_DNN'])}")
    # print(f"Length of attend_music_corrs_DNN: {len(dnn_correlations['attend_music_corrs_DNN'])}")
    print(f"Number of channels: {len(speech_music_ch_names)}")

    # Iterate through each channel name in channelnames_speech_music.txt
    for ch_idx, ch in enumerate(speech_music_ch_names):
        if ch not in ch_name_to_index:
            print(f"Channel {ch} not found in tdt_elecs_all_combined.mat for subject {subj_id}.")
            continue

        # Get the index in tdt_elecs_all_combined.mat
        idx = ch_name_to_index[ch]

        # Get anatomical label and channel name
        anat = anatomy_labels[idx]
        if isinstance(anat, (np.ndarray, list)):
            anat = anat[0]

        # Map anatomical label to short anatomical label
        short_anat = 'other'
        for roi in all_rois:
            if roi['anat_label'] == anat:
                short_anat = roi['short_anat']
                break

        # Get electrode coordinates
        x, y, z = elecmatrix[idx]

        # Handle TCH5 specifically: skip if correlations are missing
        if subj_id == 'TCH5' and ch_idx >= len(dnn_correlations['speech_music_corrs_DNN']):
            print(f"Skipping channel {ch_idx} for TCH5 (missing speech_music correlation).")
            continue

        # Append row to the dataframe only if the channel has correlations
        if ch_idx < len(dnn_correlations['speech_only_corrs_DNN']):
            row = {
                'subj_id': subj_id,
                'channel': ch_idx,  # Index from speech_music.txt
                'anat': anat,
                'channelnames': ch,
                'short_anat': short_anat,
                'speech_only_corrs_DNN': dnn_correlations['speech_only_corrs_DNN'][ch_idx],
                'music_only_corrs_DNN': dnn_correlations['music_only_corrs_DNN'][ch_idx],
                'speech_music_corrs_DNN': dnn_correlations['speech_music_corrs_DNN'][ch_idx] if ch_idx < len(dnn_correlations['speech_music_corrs_DNN']) else np.nan,
                'stacked_corrs_DNN': dnn_correlations['stacked_corrs_DNN'][ch_idx] if ch_idx < len(dnn_correlations['stacked_corrs_DNN']) else np.nan,
                # 'attend_speech_corrs_DNN': dnn_correlations['attend_speech_corrs_DNN'][ch_idx] if ch_idx < len(dnn_correlations['attend_speech_corrs_DNN']) else np.nan,
                # 'attend_music_corrs_DNN': dnn_correlations['attend_music_corrs_DNN'][ch_idx] if ch_idx < len(dnn_correlations['attend_music_corrs_DNN']) else np.nan,
                'age': age,
                'dev': dev_stage,
                'sex': sex,
                'site': site,
                'x': x,
                'y': y,
                'z': z
            }
            rows.append(row)

    return pd.DataFrame(rows)

def main_dnn_analysis():
    print("Starting DNN analysis")

    # Load all paths
    base_dir = '/Users/rajviagravat'
    participant_info_path = os.path.join(base_dir, 'Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/participant_info.xlsx')
    participant_info = pd.read_excel(participant_info_path)
    save_csv_dir = os.path.join(base_dir, 'Documents/Hamilton_Lab/Code/speechMusic/analysis/plotting/DNN_analysis')

    all_subject_dfs = []

    # Loop through each subject
    for subj_id in participant_info['subj_id']:
        print(f"Processing subject: {subj_id}")

        # Define paths for ECoG and STRF files (check both ECoG_Backup and TCH_ECoG directories)
        ecog_speechmusic_paths = [
            f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{subj_id}_ECoG_speechmusic.hf5',
            f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{subj_id}_ECoG_speechmusic.hf5'
        ]
        ecog_movietrailers_paths = [
            f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/{subj_id}_ECoG_matrix.hf5',
            f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/{subj_id}_ECoG_matrix.hf5'
        ]

        strf_speech_paths = [
            f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/strfs_speechmusic/STRF_by_spec_MT_speech.hf5',
            f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/strfs_speechmusic/STRF_by_spec_MT_speech.hf5'
        ]
        strf_music_paths = [
            f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/strfs_speechmusic/STRF_by_spec_MT_music.hf5',
            f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/strfs_speechmusic/STRF_by_spec_MT_music.hf5'
        ]
        strf_speechmusic_paths = [
            f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/strfs_og/STRF_by_spec_MT.hf5',
            f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/strfs_og/STRF_by_spec_MT.hf5'
        ]

        # Subjects that use shifted STRF files
        shifted_subjects = ['TCH13', 'TCH14', 'TCH15', 'TCH16', 'TCH18', 'TCH19', 'TCH20', 'TCH22']

        if subj_id in shifted_subjects:
            strf_stacked_paths = [
                f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/strfs_stacked_speechmusic/STRF_by_stacked_speechmusic_spec_shifted.hf5',
                f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/strfs_stacked_speechmusic/STRF_by_stacked_speechmusic_spec_shifted.hf5'
            ]
        else:
            strf_stacked_paths = [
                f'{base_dir}/Library/CloudStorage/Box-Box/ECoG_Backup/sub-{subj_id}/strfs_stacked_speechmusic/STRF_by_stacked_speechmusic_spec.hf5',
                f'{base_dir}/Library/CloudStorage/Box-Box/TCH_ECoG/sub-{subj_id}/strfs_stacked_speechmusic/STRF_by_stacked_speechmusic_spec.hf5'
            ]

        # Find the first valid path for each file
        ecog_speechmusic_path = next((path for path in ecog_speechmusic_paths if os.path.exists(path)), None)
        ecog_movietrailers_path = next((path for path in ecog_movietrailers_paths if os.path.exists(path)), None)
        strf_speech_path = next((path for path in strf_speech_paths if os.path.exists(path)), None)
        strf_music_path = next((path for path in strf_music_paths if os.path.exists(path)), None)
        strf_speechmusic_path = next((path for path in strf_speechmusic_paths if os.path.exists(path)), None)
        strf_stacked_path = next((path for path in strf_stacked_paths if os.path.exists(path)), None)
        # strf_attend_speech_path = next((path for path in strf_attend_speech_paths if os.path.exists(path)), None)
        # strf_attend_music_path = next((path for path in strf_attend_music_paths if os.path.exists(path)), None)

        # Determine site based on which path was found
        if ecog_speechmusic_path and 'ECoG_Backup' in ecog_speechmusic_path:
            site = 'ECoG_Backup'
        elif ecog_speechmusic_path and 'TCH_ECoG' in ecog_speechmusic_path:
            site = 'TCH_ECoG'
        else:
            site = 'Unknown'

        # Check if all required files exist and report missing files
        missing_files = []
        if not ecog_speechmusic_path:
            missing_files.append(f"ECoG speechmusic file: {ecog_speechmusic_paths}")
        if not ecog_movietrailers_path:
            missing_files.append(f"ECoG movietrailers file: {ecog_movietrailers_paths}")
        if not strf_speech_path:
            missing_files.append(f"STRF speech file: {strf_speech_paths}")
        if not strf_music_path:
            missing_files.append(f"STRF music file: {strf_music_paths}")
        if not strf_speechmusic_path:
            missing_files.append(f"STRF speechmusic file: {strf_speechmusic_paths}")

        if missing_files:
            print(f"Skipping subject {subj_id} due to missing files:")
            for missing_file in missing_files:
                print(f"  - {missing_file}")
            continue

        # Perform DNN analysis (stacked and attention paths can be None)
        (speech_only_corrs_DNN, music_only_corrs_DNN, speech_music_corrs_DNN, 
         stacked_corrs_DNN, attend_speech_corrs_DNN, attend_music_corrs_DNN) = dnn_analysis(
            subj_id, ecog_speechmusic_path, ecog_movietrailers_path, strf_speech_path, 
            strf_music_path, strf_speechmusic_path, strf_stacked_path,
            # strf_attend_speech_path, strf_attend_music_path
        )

        # Load anatomy data (check both ECoG_imaging and TCH_imaging directories)
        anatomy_paths = [
            os.path.join(base_dir, f'Library/CloudStorage/Box-Box/ECoG_imaging/{subj_id}_complete/elecs/tdt_elecs_all_combined.mat'),
            os.path.join(base_dir, f'Library/CloudStorage/Box-Box/TCH_imaging/{subj_id}_complete/elecs/tdt_elecs_all_combined.mat')
        ]
        anatomy_data = None
        for path in anatomy_paths:
            if os.path.exists(path):
                anatomy_data = loadmat(path)
                break

        if anatomy_data is None:
            print(f"Skipping subject {subj_id} due to missing anatomy data.")
            continue

        # Create DNN dataframe
        dnn_correlations = {
            'speech_only_corrs_DNN': speech_only_corrs_DNN,
            'music_only_corrs_DNN': music_only_corrs_DNN,
            'speech_music_corrs_DNN': speech_music_corrs_DNN,
            'stacked_corrs_DNN': stacked_corrs_DNN,
            # 'attend_speech_corrs_DNN': attend_speech_corrs_DNN,
            # 'attend_music_corrs_DNN': attend_music_corrs_DNN
        }
        df = create_dnn_dataframe(subj_id, anatomy_data, participant_info, dnn_correlations, site)
        all_subject_dfs.append(df)

    # Concatenate all subject dataframes and save to CSV
    if all_subject_dfs:
        final_df = pd.concat(all_subject_dfs, ignore_index=True)
        csv_filename = f'DNN_analysis_allmodels_{today_datetime}.csv'
        csv_save_path = os.path.join(save_csv_dir, csv_filename)
        final_df.to_csv(csv_save_path, index=False)
        print(f"Saved DNN analysis results to {csv_save_path}")
    else:
        print("No data to save.")

# Run the main DNN analysis function
if __name__ == "__main__":
    main_dnn_analysis()