# Load the stimulus and response data
import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
from mne import io
import numpy as np
from numpy.polynomial.polynomial import polyfit
from audio_tools import spectools, fbtools, phn_tools
from scipy.io import wavfile
from scipy.signal import hann, spectrogram, resample, hilbert, butter, filtfilt, boxcar, convolve
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
import glob
import re
import textgrid 
import csv
from urllib.parse import quote

from matplotlib import pyplot as plt
import parselmouth as pm
from parselmouth.praat import call
import pandas as pd
import textgrid as tg
from praatio import tgio

# import pandas as pd 

#import functions from additional script 
from ECoG_phn_alignment_tools import *

def binary_phn_mat_stim(user, stimulus_class, block, wav_name, ep, data_dir, sfreq=100.0): #get rid of looping through basename
	if stimulus_class == 'TIMIT':
		#TIMIT basename:
#         event_file = '%s/sub-%s/%s_B%d/%s_B%d_%s%d_all_events.txt'%(event_file_dir, subject,subject, block, subject, block, stimulus_class, timit_block)
		#event_file = '%s/%s_B%d/%s_B%d_%s_events.txt' %(data_dir, subject, block, subject, block, timit_block)
		event_file = f'{data_dir}/{block}/{block}_sentence-eve.txt' #txt file with raw.sfreq sampling rate already - no conversion needed
 
		evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1,2)) #read timing of events
		evs[:,:2] = evs[:,:2]
		evs = evs.astype(int) #convert these seconds into samples 
		dataframe = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe.csv')
		#evnames = np.loadtxt(event_file, dtype=str, usecols = (3)) #name of all TIMIT wav files 
		evnames = np.array(dataframe['name']+'.wav').astype('str')
		evs_orig = evs.copy()
		#print(evs)

		#read into TIMIT phoneme file:
		#TIMIT_file ='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/event_files/TIMIT_phn_info_index.txt'
		#TIMIT_file = f'{data_dir}/{block}/{block}_phn_event_times_dataframe.csv' 
		TIMIT_file = f'/Users/{user}/Library/CloudStorage/Box-Box/Stimuli/TIMIT_phn_info_index.txt'
		time_samples = np.loadtxt(TIMIT_file, dtype=float, usecols=(2))
		sentence_name = np.loadtxt(TIMIT_file, dtype=str, usecols = (5))
		phoneme_cat = np.loadtxt(TIMIT_file, dtype=str, usecols = (4))
		phoneme = np.loadtxt(TIMIT_file, dtype=str, usecols=(3))

		# phn_df = pd.read_csv(TIMIT_file)
		# time_samples = np.array(phn_df['onset_time'] *sfreq)#onset time2
		
		# #time_samples = (time_samples/128.0)*sfreq #these were in 128Hz EEG sampling so need to convert timings to 100Hz ECoG
		# phoneme = np.array(phn_df['phn']).astype('str') #phoneme
		#phoneme_cat = np.loadtxt(TIMIT_file, dtype=str, usecols = (4)) #phoneme category
		#sentence_name = np.loadtxt(TIMIT_file, dtype=str, usecols = (5)) #name of TIMIT stimuli/sentence read from 
		#event_phn = phn_df['event_id']
#         sentence_name = np.array([ name +'.wav' for name in sentence_name])
		#print(time_samples)
		
	else:
		#automatically read into MT phoneme file 
		#event_file = '%s/%s/audio/%s_%s_events.txt'%(event_file_dir, subject,subject, stimulus_class)
		#event_file = '%s/%s_B%d/%s_B%d_trailer_eve.txt' %(data_dir, subject, block, subject, block)
		event_file = f'{data_dir}/{block}/{block}_sentence-eve.txt' #txt file with raw.sfreq sampling rate already - no conversion needed
 
		evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1,2)) #read timing of events
		evs[0:2] = evs[0:2]
		evs = evs.astype(int) #convert these seconds into samples 
		#evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
		dataframe = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe.csv')
		#evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
		evnames = np.array(dataframe['name']+'.wav').astype('str')
		evs_orig = evs.copy()

		#read into MT phoneme file:
		#trailer_file ='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/event_files/trailer_phn_info_index.txt'
		trailer_file = f'/Users/{user}/Library/CloudStorage/Box-Box/Stimuli/trailer_phn_info_index.txt'

		phoneme = np.loadtxt(trailer_file, dtype=str, usecols=(0)) #phoneme
		time_samples = np.loadtxt(trailer_file, dtype=int, usecols = (1)) #name of TIMIT stimuli/sentence read from 
		time_samples = (time_samples/128.0)*sfreq
		phoneme_cat = np.loadtxt(trailer_file, dtype=str, usecols = (2)) #phoneme category
		sentence_name = np.loadtxt(trailer_file, dtype=str, usecols = (3)) #name of movie trailer stim
	
	#assign to new variable 
	phn_seconds = time_samples

	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 
	'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 
	'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 
	't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
	
	assign_num = {i: idx for idx, i in enumerate(phn1)}
	idx_num = [assign_num[i] for i in phn1]

	timing = dict()
	binary_phn_mat = dict()

	timing[wav_name] = []
	mat_length = ep.shape[2]
	print(mat_length)
	binary_phn_mat = np.zeros((len(np.unique(phn1)), mat_length))
	print(binary_phn_mat.shape)
	
	for i, s in enumerate(sentence_name):
		# print(wav_name, s)
		if s == wav_name:
			phn_time = time_samples[i]
			phn_time = int(phn_time)
			timing[wav_name].append(phn_time)
			timit_phn = phoneme[i]
			
			if timit_phn in phn1:
				phoneme_idx = assign_num[timit_phn]
				
				binary_phn_mat[phoneme_idx, phn_time] = 1

	binary_feat_mat, fkeys = phn_tools.convert_phn(binary_phn_mat.T, 'features')

		
	return binary_feat_mat.T, binary_phn_mat

def make_envelopes(wav_dir, wav_name, new_fs, ep, pad_next_pow2=True):    

	print("Sentence: %s"% (wav_name))
	wfs, sound = wavfile.read('%s/%s'%(wav_dir, wav_name))
	sound = sound/sound.max()
	#all_sounds[wav_name] = sound
	envelopes = []


	envelope = spectools.get_envelope(sound, wfs, new_fs, pad_next_pow2=pad_next_pow2)
	
	return envelope


def stimuli_mel_spec(path, wav_name, new_fs):
	[fs,w] = wavfile.read(path+'/'+ wav_name)
	w=w.astype(float)
	
	mel_spec, freqs = spectools.make_mel_spectrogram(w, fs, wintime=0.025, steptime=1/new_fs, nfilts=80, minfreq=0, maxfreq=None)

	return mel_spec, freqs


def get_meanF0s_v2(fileName, steps=1/128.0, f0min=50, f0max=300):
	"""
	Uses parselmouth Sound and Pitch object to generate frequency spectrum of
	wavfile, 'fileName'.  Mean F0 frequencies are calculated for each phoneme
	in 'phoneme_times' by averaging non-zero frequencies within a given
	phoneme's time segment.  A range of 10 log spaced center frequencies is
	calculated for pitch classes. A pitch belongs to the class of the closest
	center frequency bin that falls below one standard deviation of the center
	frequency range.
	
	"""
	#fileName = wav_dirs + wav_name
	sound =  pm.Sound(fileName)
	pitch = sound.to_pitch(steps, f0min, f0max) #create a praat pitch object
	pitch_values = pitch.selected_array['frequency']
	
	return pitch_values  


#more pitch functions make a 1D pitch vector into 10 pitch features to see if there is specific electrode tuning
def get_bin_edges_percent_range(a, bins=10, percent=95):
	assert percent > 1 
	assert percent < 100
	tail_percentage = (100 - percent)/2
	a_range = np.percentile(a, [tail_percentage, 100-tail_percentage])
	counts, bin_edges = np.histogram(a, bins=bins, range=a_range)
	return bin_edges

def get_pitch_matrix(pitch, bin_edges):
	pitch[pitch < bin_edges[0]] = bin_edges[0] + 0.0001
	pitch[pitch > bin_edges[-1]] = bin_edges[-1] - 0.0001
	bin_indexes = np.digitize(pitch, bin_edges) - 1
	stim_pitch = np.zeros((len(pitch), 10))
	for i, b in enumerate(bin_indexes):
		if b < 10:
			stim_pitch[i, b] = 1
	return stim_pitch


def trailer_soundCat_matrix(nat_sound_event_dir, wav_name, evs, epochs, show_fig=False):
	sc_dict = dict()
	timing = dict()

	category = ['ES', 'FS', 'IM', 'SO', 'IMV', 'MS', 'ENS', 'AVS', 'NVS', 'NS', 'NSV']

	assign_num = {i: idx for idx, i in enumerate(np.unique(category))}
	idx_num = [assign_num[i] for i in category]

	name = wav_name.replace('.wav', '') + '_corrected_natsounds.txt'

	#load timing info, category, and event_id from saved textfiile
	onset = np.loadtxt(f'{nat_sound_event_dir}/{name}', dtype=float, usecols = (0))
	offset = np.loadtxt(f'{nat_sound_event_dir}/{name}', dtype=float, usecols = (1))
	category_txt = np.loadtxt(f'{nat_sound_event_dir}/{name}', dtype=str, usecols = (2))
	#event_id = np.loadtxt(f'{nat_sound_event_dir}/{name}', dtype=float, usecols = (3))
	sentence_name = np.loadtxt(f'{nat_sound_event_dir}/{name}', dtype=str, usecols = (4))
	print(onset.max())
	#now align event information with timing information from ECoG block
	aligned_onset = onset
	#     aligned_offset = offset +evs[0][0]



	timing = dict()
	cat_mat = dict()

	timing[wav_name] = []
	mat_length = epochs.shape[2]
	print(mat_length)

	full_cat_mat = np.zeros((len(np.unique(category)), mat_length))
	print(full_cat_mat.shape)

	for i, s in enumerate(sentence_name):
		if s == wav_name:
			categorization_time = aligned_onset[i]
			categorization_time = int(categorization_time)
			timing[wav_name].append(categorization_time)
			category_label = category_txt[i]

			if category_label in np.unique(category):
				index_cat = assign_num[category_label]
				full_cat_mat[index_cat, categorization_time] = 1

	if show_fig:
		plt.imshow(full_cat_mat, interpolation='nearest', aspect='auto')
		plt.gca().set_yticks(np.arange(len(np.unique(category))))
		plt.gca().set_yticklabels(np.unique(category))
		
	return full_cat_mat

def scene_cut_feature(wav_name, textgrid_dir, fs=100.0):
	'''
	Uses the textgrid scene cuts for movie trailers and creates a binary matrix for 
	where a scene cut takes place. 

	Appends instances of scene cuts in samples (fs=128.0) to full_AV_matrix which contains 
	stim and resp for all subjects with corresponding auditory (and visual for MTs) features
	'''
	#with h5py.File(f'{h5_dir}/{subject}_ECoG_matrix.hf5', 'a') as g:

	#for name in glob.glob('%s/*.TextGrid' %(textgrid_dir)):
	wav_name = wav_name.replace('.wav', '')
	name = f'{textgrid_dir}/{wav_name}_corrected_SC.TextGrid'
	print(name)
	r = open(name)
	grid = tg.TextGrid(r.read())
	tier_names = [t.nameid for t in grid.tiers]
	for r, i in enumerate(tier_names):
		if i == 'Scene cut':
			print(r)

	scene_tier = [t for t in grid.tiers[r].simple_transcript]
	scene_tier = np.array(scene_tier)

	scene_onset = scene_tier[:,0].astype(float)*fs
	scene_onset = scene_onset.astype(int)
	scene_offset = scene_tier[:,1].astype(float)*fs
	scene_offset = scene_offset.astype(int)

	name = name.replace(textgrid_dir, '')
	name = name.replace('_SC.TextGrid', '.wav')
	name = name.replace('_corrected', '')
	name = name.replace('/', '')
	print(name)

	#create matrix length of trailer and append scene onset as 1 
	matrix_values = np.zeros((scene_offset[-1],))
	for i in scene_onset:
		matrix_values[i] = 1 
	print(matrix_values.shape)
	matrix_dims = np.expand_dims(matrix_values, axis=0)
	
	return matrix_dims

			#append to h5 file /stim/%s/scene_cut as binary matrix 
			#g.create_dataset('%s/%s/stim/scene_cut'%(stimulus_class,name), data=np.array(matrix_dims, dtype=float))


def get_cse_onset(specgram, audio=None, audio_fs=None, wins = [0.04], nfilts=80, pos_deriv=True, spec_noise_thresh=1.04):
	"""
	Get the onset based on cochlear scaled entropy
	Inputs:
		audio [np.array] : your audio
		audio_fs [float] : audio sampling rate
		wins [list] : list of windows to use in the boxcar convolution
		pos_deriv [bool] : whether to detect onsets only (True) or onsets and offsets (False)
	Outputs:
		cse [np.array] : rectified cochlear scaled entropy over window [wins]
		auddiff [np.array] : instantaneous derivative of spectrogram
	"""
	new_fs = 100 # Sampling frequency of spectrogram
	if audio is not None:
		specgram = spectools.get_mel_spectrogram(audio, audio_fs, nfilts=nfilts)
	else:
		print('using previously collected spectrogram')
	specgram[specgram<spec_noise_thresh] = 0
	nfilts, ntimes = specgram.shape

	if pos_deriv is False:
		auddiff= np.sum(np.diff(np.hstack((np.atleast_2d(specgram[:,0]).T, specgram)))**2, axis=0)
	else:
		all_diff = np.diff(np.hstack((np.atleast_2d(specgram[:,0]).T, specgram)))
		all_diff[all_diff<0] = 0
		auddiff = np.sum(all_diff**2, axis=0)
	cse = np.zeros((len(wins), ntimes))

	# Get the windows over which we are summing as bins, not times
	win_segments = [int(w*new_fs) for w in wins]

	for wi, w in enumerate(win_segments):
		box = np.hstack((np.atleast_2d(boxcar(w)), -np.ones((1, int(0.15*new_fs))))).ravel()
		cse[wi,:] = convolve(auddiff, box, 'full')[:ntimes]

	cse[cse<0] = 0
	cse = cse/cse.max()

	return cse, auddiff
