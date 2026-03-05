import textgrid 
import pandas as pd 
import re
import numpy as np 
import mne
from mne import io
from audio_tools import spectools, fbtools, phn_tools
from matplotlib import pyplot as plt
import os
import csv


def TIMIT_phn_groups(subject, data_dir, block, timit_block, raw=100.0):
	'''
		*** RUN THIS FUNCTION FOR EVERY ECoG BLOCK ***
		Code is here to generate textfile if needed and outputs the following information:
		1st column: File index number
		2nd column: Index of each phoneme event based on time for the given file 
		3rd column: Time in samples of where phoneme occurs
		4th column: phoneme 
		5th column: phoneme category 
		6th column: file in which phoneme + time belong to
		Returns: mydata which contains info in the textfile
	'''

	#Load event file form task so you can get the sampling rate which will be used later
	# to convert seconds into samples
	event_file = pd.read_csv(f'{data_dir}/{block}/{block}_event_times_dataframe.csv')
	#event_file = '%s/sub-%s/%s_B%d/%s_B%d_TIMIT%d_events.txt' %(data_dir, subject,subject,block,subject, block,timit_block )
	evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1,2)) #read timing of events
	evs = event_file['onset_time']
	evs[:,:2] = evs[:,:2]
	# evs = evs.astype(int) #convert these seconds into samples 
	evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
	evs_orig = evs.copy()

	basename = [w[:-4] for w in evnames]
	first_phoneme = []
	for idx, b in enumerate(basename):
		phns = np.loadtxt('%s/sub-%s/Stimuli/TIMIT/%s.phn'%(data_dir,subject,b), dtype=int, usecols = (0))
		#print(phns[1])
		phnnames = np.loadtxt('%s/sub-%s/Stimuli/TIMIT/%s.phn'%(data_dir,subject,b), dtype=np.str, usecols = (2))
		#print(phnnames[1])
		first_phoneme_time = phns[1]/16000. 
		# first_phoneme_sample = first_phoneme_sample.astype(int)
		first_phoneme.append(phnnames[1])
		evs[idx,0] = evs[idx,0] + first_phoneme_time 


	#loop through and get ALL phonemes, not just the first one:
	#Get all types of phonemes based on family/category: 
	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch', 'hv']
	plosives =['p','t','k','b','bcl','d','g', 'v', 'bcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl' ]
	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw', 'ax-h']
	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng', 'eng'] 


	all_phonemes = []

	onset = []
	phoneme = []
	file_num = [] #this is idx
	sample_num = []
	phon_cat = []
	TIMIT_name = []


	for idx, b in enumerate(basename):
		phns_onset = np.loadtxt('%s/sub-%s/Stimuli/TIMIT/%s.phn'%(data_dir, subject, b), dtype=int, usecols = (0)) #start time - samples
		phns_onset = phns_onset/16000. #load in seconds
		# phns_onset = phns_onset.astype(int)
		onset.append(phns_onset)
		phnnames_phonemes = np.loadtxt('%s/sub-%s/Stimuli/TIMIT/%s.phn'%(data_dir, subject,b), dtype=np.str, usecols = (2)) #phonemes

		phoneme.append(phnnames_phonemes)
		file_num.append([idx]*len(phnnames_phonemes))
		sample_num.append(np.arange(len(phnnames_phonemes)))

		phon_group = []
		for phon in phnnames_phonemes:

			if phon in fricatives:
				phon_group.append('fric')
				TIMIT_name.append(b)

			elif phon in plosives:
				phon_group.append('plos')
				TIMIT_name.append(b)

			elif phon in vowels:
				phon_group.append('vow')
				TIMIT_name.append(b)

			elif phon in nasals:
				phon_group.append('nas')
				TIMIT_name.append(b)

			else:
				phon_group.append('other')
				TIMIT_name.append(b)

		assert len(phnnames_phonemes) == len(phon_group), 'More labels made than samples'
		phon_cat.append(phon_group)


	phoneme = np.concatenate(phoneme)
	onset = np.concatenate(onset)
	file_num = np.concatenate(file_num)
	sample_num = np.concatenate(sample_num)
	phon_cat = np.concatenate(phon_cat)

	TIMIT_name = np.concatenate([np.expand_dims(i,axis=0) for i in TIMIT_name])

	#create array
	mydata = np.stack([file_num, sample_num, onset, phoneme, phon_cat, TIMIT_name], axis=1)

	#save contents of mydata into textfile:
	phn_text_file = '%s/sub-%s/%s_B%d/ECoG_TIMIT_info_phn.txt'%(data_dir,subject,subject,block)
	print("Saving phn_text_file: %s"%phn_text_file)
	np.savetxt('%s/sub-%s/%s_B%d/ECoG_TIMIT_info_phn.txt'%(data_dir,subject,subject,block),mydata, fmt='%s\t', delimiter='\t')
	np.savetxt('%s/sub-%s/%s_B%d/ECoG_TIMIT_info_phn.csv'%(data_dir,subject,subject,block), mydata, fmt='%s\t', delimiter='\t')
	return mydata, phon_cat


def phn_categories(data_dir, subject, block, stimuli):
	'''
	Initializing phonemes which are the same across both movie trailers and TIMIT (as shown from Liberty's bar plot)
	Assign an index number to each phoneme 
	Append the index number to each phoneme based on phoneme category
	'''

	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 
	'ih', 'iy', 'jh', 'k', 'l', 'm','n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v',
	 'w', 'y', 'z', 'zh']

	#assign index to each phoneme in phn1 list:
	assign_num = {i: idx for idx, i in enumerate(phn1)}
	idx_num = [assign_num[i] for i in phn1]

	#path:
	# datadir = '/Users/maansidesai/Box/MovieTrailersTask/Data/ECoG/Participants/sub-%s/%s_B%d' %(subject,subject,block)

	if stimuli == 'TIMIT':
		#timit_dir = '%s/Stimuli/TIMIT'%(datadir)
		read = '%s/sub-%s/%s_B%d/ECoG_TIMIT_info_phn.csv' %(data_dir, subject, subject, block)     
		reader = pd.read_csv(read,index_col=None, header=0,encoding = "ISO-8859-1")
		reader = reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
		reader.columns = ['idx_sent', 'idx_phn', 'sample', 'phn', 'phn_cat', 'sentence']

		phonemes = reader['phn']
		index = np.empty((reader.shape[0],))
		for i, phon in enumerate(reader['phn']):
			try:
				index[i] = assign_num[phon]
			except:
				index[i] = np.nan
		reader['index'] = index
		np.savetxt('%s/sub-%s/%s_B%d/ECoG_TIMIT_info_phn.txt' %(data_dir, subject, subject, block), reader , fmt='%s\t', delimiter='\t')

	elif stimuli == 'MovieTrailers':
		#trailer_dir = '%s/%s_trailer_phn_info.csv'%(datadir,subject)
		mt = '%s/ECoG_trailer_phn_info.csv' %(datadir) 
		mt_reader = pd.read_csv(mt,index_col=None, header=0,encoding = "ISO-8859-1")
		mt_reader = mt_reader.dropna(axis=1, how='all') #drop NAs that appear in columns 
		mt_reader.columns = ['phn', 'sample','phn_cat', 'trailer']

		phonemes = mt_reader['phn']

		index = np.empty((mt_reader.shape[0],))
		for i, phon in enumerate(mt_reader['phn']):
			try:
				index[i] = assign_num[phon]
			except:
				index[i] = np.nan

		mt_reader['index'] = index
		np.savetxt('%s/ECoG_trailer_phn_info.txt' %(datadir), mt_reader , fmt='%s\t', delimiter='\t')

	else:
		print('Could not detect correct input')


#function to get phoneme + timing event file for movie trailers 
def get_trailer_phns_event_file(subject, block, data_dir, raw=100.0):

	'''
		Run this function for every subject. 
		This will output the phoneme and sample (timing) info for each trailer that a subject heard/watched
		Running this function everytime is important because the subjects do not always hear/watch every trailer, 
		however they do listen to all five blocks of TIMIT 

		Code is here to generate textfile if needed and outputs the following information:

		1st column: phoneme
		2nd column: Time in samples of where phoneme occurs 
		3rd column: category of phoneme
		4th column: Name of trailer

	'''
	#datadir='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/%s/downsampled_128'%(subject)
	event_file = '%s/sub-%s/%s_B%d/%s_B%d_trailer_eve.txt'%(data_dir,subject,subject,block,subject,block)
	evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1)) #read timing of events
	event_id = np.loadtxt(event_file, dtype='f', usecols = (2))
	evs = evs[0:2]*raw
	evs = np.append (evs, [event_id])
	evs = evs.astype(int) #convert these seconds into samples 
	evnames = np.loadtxt(event_file, dtype=np.str, usecols = (3)) #name of all TIMIT wav files 
	evs_orig = evs.copy()

	basename = evnames.tolist()[:-4] # This is the name of the wav file without .wav
		#Get all types of phonemes based on family/category: 
	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch']
	plosives =['p','t','k','b','bcl','d','g', 'v']
	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw']
	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng'] 

	#Creating new categories based on phoneme features:
	
	trailer_phn_start_time = [] #start time of phoneme
	trailer_phn_event_name = [] #each phoneme from text grid transcriptions 
	trailer_name = [] #name of movie trailer that correlates with start time and phoneme 
	trailer_name2 = [] # to append all trailers based on length of phonemes 
	trailer_phon_cat = []

	tg_dir = '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/stimuli/MovieTrailers/textgrids/Corrected/'
	r = open('%s/%s_corrected.TextGrid'%(tg_dir,basename))
	tg = textgrid.TextGrid(r.read())		
	tier_names = [t.nameid for t in tg.tiers]
	print('Now reading the file: %s' %(basename))
	tier_names_nospace = [t.nameid.replace(" ", "") for t in tg.tiers]
	tier_num = 0
	all_phonemes=[t[2] for t in tg.tiers[tier_num].simple_transcript]
	all_phonemes = [x.lower() for x in all_phonemes]  #need to make all phoneme strings lower case to match TIMIT
	print("The unique phonemes are:") #gives all phonemes for each movietrailer in basename
	print(np.unique(all_phonemes))
	print('--------------------------------------')

	phon_group = []
	for phon in all_phonemes:

		if phon in fricatives:
			phon_group.append('fric')
			trailer_name.append(basename)

		elif phon in plosives:
			phon_group.append('plos')
			trailer_name.append(basename)

		elif phon in vowels:
			phon_group.append('vow')
			trailer_name.append(basename)

		elif phon in nasals:
			phon_group.append('nas')
			trailer_name.append(basename)

		else:
			phon_group.append('other')
			trailer_name.append(basename)

	assert len(all_phonemes) == len(phon_group), 'More labels made than samples'
	trailer_phon_cat.append(phon_group)
	print(phon_group)

	#loop to find any numbers attached to the phonemes and eliminate (i.e. take out 1 from uw1)
	for i, p in enumerate(all_phonemes):
		all_phonemes[i] = re.sub(r'[0-9]+', '', p)

	#converting start times from seconds to samples 
	start_times = [t[0] for t in tg.tiers[tier_num].simple_transcript]
	start_times = np.asarray(start_times, dtype=np.float32)
	start_times = start_times*raw
	start_times = start_times.astype(int)
	start_times = start_times + evs[0]

	#appending to arrays 
	trailer_phn_start_time.append(start_times)
	trailer_phn_event_name.append(all_phonemes)
	trailer_name2.append([basename]*len(all_phonemes))

		#concatenatate appended arrays (above)
	trailer_phn_event_name = np.concatenate(trailer_phn_event_name)
	trailer_phn_start_time = np.concatenate(trailer_phn_start_time)
	trailer_phon_cat = np.concatenate(trailer_phon_cat)
	print(len(trailer_name))
	trailer_name = np.concatenate([np.expand_dims(i,axis=0) for i in trailer_name])

	#stack all of the arrays and save as textfile 
	phn_sample_trailer_events = np.stack([trailer_phn_event_name, trailer_phn_start_time, trailer_phon_cat, trailer_name], axis=1)
	np.savetxt('%s/sub-%s/%s_B%d/ECoG_trailer_phn_info.txt' %(data_dir, subject, subject, block), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 
	np.savetxt('%s/sub-%s/%s_B%d/ECoG_trailer_phn_info.csv' %(data_dir, subject, subject, block), phn_sample_trailer_events , fmt='%s\t', delimiter='\t') #output textfile, contains 3 columns 

	return phn_sample_trailer_events


def get_timit_phns_event_file(subject, block, timit_block, data_dir, raw_fs=100.0): 
	#TIMIT_file ='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/event_files/TIMIT_phn_info_index.txt'
	#datapath = '/Users/maansidesai/Box/MovieTrailersTask/Data/ECoG/Participants/sub-%s/%s_B%d' %(subject,subject,block)
	TIMIT_file ='%s/%s_B%d/ECoG_TIMIT_info_phn.txt' %(data_dir, subject,block)
	print(TIMIT_file)
	subj_TIMIT = '%s/%s_B%d/%s_B%d_%s_events.txt'%(data_dir,subject,block, subject,block,timit_block)
	#ddir='/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/'
	
	#read into TIMIT_file: this is TIMIT_phn_info_index.txt (large textfile you only run once)
	sentence_idx = np.loadtxt(TIMIT_file, dtype=int, usecols = (0))
	# print("unique sentence_idx:")
	# print(len(np.unique(sentence_idx)))


	time_seconds = np.loadtxt(TIMIT_file, dtype=np.float, usecols = (2))
	phoneme = np.loadtxt(TIMIT_file, dtype=np.str, usecols = (3))
	phoneme_cat = np.loadtxt(TIMIT_file, dtype=np.str, usecols = (4))
	sentence_name = np.loadtxt(TIMIT_file, dtype=np.str, usecols = (5))
	phoneme_idx = np.loadtxt(TIMIT_file, dtype=np.float, usecols = (6))
	
	onset_timit = np.loadtxt(subj_TIMIT, dtype=np.float, usecols = (0))
	offset_timit = np.loadtxt(subj_TIMIT, dtype=np.float, usecols = (1))
	timit_idx = np.loadtxt(subj_TIMIT, dtype=int, usecols = (2))
	sentence = np.loadtxt(subj_TIMIT, dtype=np.str, usecols = (3))

	#get rid of .wav from sentence (which is the variable in subject_TIMIT_all_events.txt)
	new_sentence = np.char.strip(np.unique(sentence), '.wav')

	new_onsets = []
	new_phonemes = []
	new_phonemes_cat = []
	new_sentence_name = []
	new_phoneme_idx = []
	new_sentence_idx = []
	
	# print("unique onset_timit:")
	# print(len(onset_timit))

	for ix, onset in enumerate(onset_timit):
		new_onsets.append(time_seconds[sentence_idx==ix]  + onset)
		new_phonemes.append(phoneme[sentence_idx==ix])
		new_phonemes_cat.append(phoneme_cat[sentence_idx==ix])
		new_sentence_name.append(sentence_name[sentence_idx==ix])
		new_phoneme_idx.append(phoneme_idx[sentence_idx==ix])
		new_sentence_idx.append(sentence_idx[sentence_idx==ix])
		
	new_onsets = (np.concatenate(new_onsets)*raw_fs).astype('int') #this converts to 100Hz for ECoG
	new_phonemes = np.concatenate(new_phonemes)
	new_phonemes_cat = np.concatenate(new_phonemes_cat)
	new_sentence_name = np.concatenate(new_sentence_name)
	new_phoneme_idx = np.concatenate(new_phoneme_idx)
	new_sentence_idx = np.concatenate(new_sentence_idx)

	
	#write new timing onset to new file
	new_file_out = np.stack([new_phonemes, new_onsets, new_phonemes_cat,  new_sentence_name, new_phoneme_idx], axis=1)
	# print(new_file_out)

	#save to new textfile:

	np.savetxt('%s/%s_B%d_TIMIT_phn_info.txt' %(data_dir, subject, block), new_file_out, fmt='%s\t', delimiter='\t')
	return new_file_out

def get_ECoG_trailer_file(subject, data_dir, block, sfreq=100.0):
	'''
	Load the preprocessed ECoG (using high gamma data), removing bad time points if they exist
	Input : 
		subject [str] : the subject ID (e.g. 'S0003')
		block [int] : the block for the designated task (e.g. 5)
		sfreq [int] : frequency sampling rate which you downsampled to, a standard value of 100.0
	Output : 
		raw [mne Raw object] : the data structure containing the ECoG data for this participant
	'''
	# data_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/ECoG/Participants/'
	event_phoneme_file_name = '%s/%s_B%d/%s_B%d_trailer_phoneme_eve.txt'%(data_dir, subject, block, subject, block)

	this_event = []
	event_str = []
	phon_cat = []
	phn_feat_name = []

	with open(event_phoneme_file_name, 'r') as my_csv:            # read the file as my_csv
		csvReader = csv.reader(my_csv, delimiter='\t')  # using the csv module to write the file
		for row in csvReader:
			this_event.append(row[:3])
			event_str.append(row[3])
	this_event = np.array(this_event, dtype=np.float)
	#print(this_event)
	event_samples = this_event.copy() # Make a copy of the variable first
	event_samples[:,:2] = np.round(this_event[:,:2]*sfreq)

	# Convert to integers
	event_samples = event_samples.astype(int)   
	#print(event_samples)
	unique_evs = np.unique(event_str)

	# for i, e in enumerate(unique_evs):
	# 	print("%d : %s"%(i,e))

	event_str = [x.lower() for x in event_str]
	#loop through and get ALL phonemes, not just the first one:
	#Get all types of phonemes based on family/category: 
	fricatives = ['f','v','th','dh','s','sh','z','zh','hh', 'ch', 'hv', 'h']
	plosives =['p','t','k','b','bcl','d','g', 'v', 'bcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl' ]
	vowels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'eh', 'ey', 'ih', 'ow', 'iy', 'oy', 'uh', 'uw', 'ax-h']
	nasals = ['m', 'n', 'r', 'l', 'y', 'w', 'er', 'ng', 'eng'] 

	phon_group = []
	for phon in event_str:

		if phon in fricatives:
			phon_group.append('fric')
			phn_feat_name.append(phon)

		elif phon in plosives:
			phon_group.append('plos')
			phn_feat_name.append(phon)

		elif phon in vowels:
			phon_group.append('vow')
			phn_feat_name.append(phon)

		elif phon in nasals:
			phon_group.append('nas')
			phn_feat_name.append(phon)

		else:
			phon_group.append('other')
			phn_feat_name.append(phon)

	assert len(event_str) == len(phon_group), 'More labels made than samples'
	phon_cat.append(phon_group)
	print('generated event file for movie trailer')

	return unique_evs, phon_cat


#load ECoG ICA data

def load_raw_ECoG(block, data_dir, band:str):
	'''
	band : string
		- beta_15to30_7band
		- gamma_30to70_8band
		- HilbAA_70to150_8band #for high gamma
	'''
	if band == 'high_gamma':
		print('loading high gamma neural data, oscillatory info between 70-150 Hz')
		ecog_file = '%s/%s/HilbAA_70to150_8band/ecog_hilbAA_70to150_8band_notch_car_log.fif'%(data_dir, block)
	# elif band == 'gamma': 
	# 	print('loading gamma neural data, oscillatory info between 30-70 Hz')
	# 	ecog_file = '%s/%s/HilbAA_gamma_30to70_8band/ecog_hilbAA_30to70_8band_notch_car_log.fif'%(data_dir, block)
	# elif band == 'beta':
	# 	print('loading beta neural data, oscillatory info between 15-30 Hz')
	# 	ecog_file = '%s/%s/HilbAA_beta_15to30_7band/ecog_hilbAA_15to30_7band_notch_car_log.fif'%(data_dir, block)
	raw = mne.io.read_raw_fif(ecog_file, preload=True)

	# Print which are the bad channels, but don't get rid of them yet...
	raw.pick_types(eeg=True, meg=False, exclude=[])
	bad_chans = raw.info['bads']
	print("Bad channels are: ")
	print(bad_chans)

	# Get onset and duration of the bad segments in samples
	bad_time_onsets = raw.annotations.onset * raw.info['sfreq']
	bad_time_durs = raw.annotations.duration * raw.info['sfreq']

	print(raw._data.shape)

	# Set the bad time points to zero
	for bad_idx, bad_time in enumerate(bad_time_onsets):
		raw._data[:,int(bad_time):int(bad_time+bad_time_durs[bad_idx])] = 0
	# print('********************************')
	# print('The bad time points are: %s' %(bad_time_onsets))

	
	return raw

def get_event_epoch(raw, evs, event_id, bef_aft=[0,0], baseline = None, reject_by_annotation=False, trailers=True):

	if trailers:
		#max_samp_dur = np.max(evs[(np.where(evs[2] == event_id)),1]-evs[(np.where(evs[2] == event_id)),0])
		#max_samp_dur = evs[0][1] - evs[0][0]
		max_samp_dur = evs[0][1] - evs[0][0]

		#print(evs[(np.where(evs[2] == event_id)),1]-evs[(np.where(evs[2] == event_id)),0])
	else:
		max_samp_dur = event[1] - event[0]
		# max_samp_dur = np.max(evs[(np.where(evs[:,2] == event_id)),1]-evs[(np.where(evs[:,2] == event_id)),0])
		# print(evs[(np.where(evs[:,2] == event_id)),1]-evs[(np.where(evs[:,2] == event_id)),0])


	trial_dur = max_samp_dur/raw.info['sfreq']
	print("Trial duration: %2.2f seconds"%trial_dur) 

	epochs = mne.Epochs(raw, evs, event_id=[event_id], tmin=bef_aft[0], tmax=trial_dur+bef_aft[1], baseline=baseline,
							reject_by_annotation=reject_by_annotation)
	ep = epochs.get_data()
		
	return ep


def load_event_file(event_file_name, subject, block, timit_block,
					data_dir,
					fs = 100.0):
	'''
	Get the TIMIT or MovieTrailers event file with start and stop times relative to the EEG start time for a 
	particular subject. This event file has one row per TIMIT sentence or one row per MovieTrailer, depending
	on which stimulus is chosen.
	Inputs:
		event_file_name [str] : 'TIMIT' or 'MovieTrailers' -- which event file to load
		subject [str] : subject ID (e.g. 'S0003')
		data_dir [str] : path to your data and event files
		fs [int] : the sampling frequency for returning the samples
	Output:
		evs : the offset and onset samples (assuming [fs] sampling frequency)
		wav_id : the name of the .wav that corresponds to evs
	'''
	if event_file_name=='TIMIT':
		#event_file = '%s/%s/audio/%s_TIMIT_all_events.txt'%(data_dir, subject, subject)
		event_file = '%s/%s_B%d/%s_B%d_%s_events.txt' %(data_dir, subject, block, subject, block, timit_block)
		# Load the columns with the times    
		evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1, 2))
		evs[:,:2] = evs[:,:2]*fs #100 is the downsampled frequency from ECoG data
		evs = evs.astype(int) #this takes into account onset and offset times
		wav_id = np.loadtxt(event_file, dtype='<U', usecols = 3) #name of .wav filename
		
	else:
		#event_file = '%s/%s_B%d_trailer_eve.txt' %(data_dir, block, subject, block)
		event_file = '%s/%s_B%d/%s_B%d_trailer_eve.txt' %(data_dir, subject, block,subject, block)
		evs = np.loadtxt(event_file, dtype='f', usecols = (0, 1))
		event_id = np.loadtxt(event_file, dtype='f', usecols = (2))
		evs = evs[0:2]*fs

		evs = np.append (evs, [event_id])
		evs = evs.astype(int) #this takes into account onset and offset times


		wav_id = np.loadtxt(event_file, dtype='<U', usecols = 3) 
	print(event_file)
		

	
	return evs, wav_id
