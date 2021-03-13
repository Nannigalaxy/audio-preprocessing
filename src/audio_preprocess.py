# @author: Nandan M
# @time: 31/01/2021 08:02 PM 

import os
import numpy as np
import pandas as pd
import librosa
# to write audio file output 
# from scipy.io.wavfile import write

# deterministic random generator
np.random.seed(2)


def load_audio_data(data_path, samples, seconds, sampling_rate, background):
	'''
	function to load data from each folder and as classnames
	'''
	limit=samples # samples limit per class
	data = []
	files = []
	labels = []
	category = []
	path = data_path # dataset path 
	dirs = os.listdir(path)
	dirs.sort()
	# read all directory except background noise(.background) directory if false
	if background:
		# load n second length background noise audio and split to desired length 
		bg_data = load_bg(path, sampling_rate)
		bg_chunks = split_bg(bg_data, seconds, sampling_rate)
		bg_chunks = bg_chunks[:limit]
		labels.extend([0]*len(bg_chunks))
		data.extend(bg_chunks.tolist())
		category.extend(["background"]*len(bg_chunks))
		files.extend(['N/A']*len(bg_chunks))
	t=0
	for folder in dirs[1:]:
		c=0
		for filename in os.listdir(path+folder):
			l = np.zeros(len(dirs))
			ind = dirs.index(folder) 
			if filename.endswith("wav"):
				l[ind] = 1
				labels.append(ind)
				category.append(folder)
				wave, _ = librosa.load(path + folder + "/" + filename,\
						mono=True,\
						sr=sampling_rate)
				data.append(wave)
				files.append(folder + "/" + filename)
				c+=1; t+=1
			if c==limit:
				break
		print(folder,":", c)
	print('Total original :', t)
	return data, files, labels, category

def audio_length(max_length, data):
	'''
	limit audio length to desired length
	'''
	if len(data)>max_length:
		data = data[:max_length]
	else:
		data = np.pad(data, (0, max(0, max_length - len(data))), "constant")
	return data


def split_bg(bg, seconds, sampling_rate):
	'''
	split n length bg audio files into desired chunks  
	'''
	max_length = seconds * sampling_rate
	bg_split = []
	for x in range(len(bg)):
		for i in range(0, len(bg[x]), max_length):
			chunk = bg[x][i:i + max_length]
			# print(len(chunk))
			bg_split.append(chunk)

			# create folders manually and save audio chunks 
			# write("./export/bg/audio"+str(x)+str(i)+".wav", sr, chunk)

	return np.array(bg_split)


def load_bg(path, sampling_rate):
	'''
	load background for augmentation
	'''
	bg = []
	# hardcoded background directory name
	folder = '.background'
	for filename in os.listdir(path+folder):
		if filename.endswith("wav"):
			wave, _ = librosa.load(path + folder + "/" + filename,\
					mono=True,\
					sr=sampling_rate)
			bg.append(wave)
	print('\nBackground :', len(bg))
	return bg

def get_random_bg(bg, max_length):
	'''
	select random background of desired length
	'''
	index = np.random.randint(0,len(bg))
	bg_audio = bg[index]
	i = np.random.randint(0,bg_audio.shape[0]-max_length)
	bg_chunk = bg_audio[i:i + max_length]
	return bg_chunk

# Augmentation functions
def change_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

def shift_audio(data, sampling_rate, shift_max, shift_direction):
	shift = np.random.randint(0,sampling_rate * shift_max)
	if shift_direction == 1: # right
		shift = -shift

	augmented_data = np.roll(data, shift)
	# Set to silence for heading/ tailing
	if shift > 0:
		augmented_data[:shift] = 0
	else:
		augmented_data[shift:] = 0
	return augmented_data


def augment_audio(df_row, bg, sampling_rate, max_length):
	''' 
	augment voice audio and overlap with background
	'''
	bg_audio = get_random_bg(bg, max_length)
	audio = np.array(df_row.audio)
	# audio.resize(bg_audio.shape)

	# shift time left or right
	shift_time = np.random.randint(1, 50)*0.01 # shift range 0.01 - 0.50 secs
	shift_direction = np.random.randint(0, 2) # 0: left, 1: right
	audio = shift_audio(audio, sampling_rate, shift_time, shift_direction)
	# using shift_direction for more random augmentation activation
	if shift_direction==1:
		pitch_factor = np.random.randint(-30, 30)*0.1 
		audio = change_pitch(audio, sampling_rate, pitch_factor)
	else:
		speed_factor = np.random.randint(7, 13)*0.1 #  If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
		audio = change_speed(audio, speed_factor)

	# audio intensity for background and voice
	bg_vol = np.random.randint(10, 40)*0.01 # bg volume range 0.10 - 0.40
	voice_vol = np.random.randint(60, 100)*0.01 # voice volume range 0.60 - 1.00
	audio = audio_length(max_length, audio)
	aug_audio = np.array(np.add(bg_vol*bg_audio, voice_vol*audio))

	return aug_audio

def audio_synthesis(path, sampling_rate, sample_limit, seconds, max_length, random_factor, background):
	'''
	Load raw dataset and background
	'''
	audio_data, files, label, category = load_audio_data(path, sample_limit, seconds, sampling_rate, background)
	data_dict = {'file':files,'audio':audio_data,'label':label,'category':category}
	dataset = pd.DataFrame(data_dict)
	classname = dataset.category.unique()
	bg = load_bg(path, sampling_rate)

	n_list = []
	r = 0

	for i in range(dataset.shape[0]):
		row = dataset.iloc[i]
		# random number of samples augmentation
		rand_f = np.random.randint(2, random_factor) # higher the max limit more chances of augmented audio
		lim_audio = audio_length(max_length, row.audio)
		n_list.append([row.file,lim_audio,row.label,row.category])
		# write("./export/"+str(row.category)+"/audio"+str(i)+".wav", sampling_rate, lim_audio)
		for x in range(rand_f):
			aug_audio = augment_audio(row, bg, sampling_rate, max_length)
			lim_audio = audio_length(max_length, aug_audio)
			n_list.append([row.file,lim_audio,row.label,row.category])
			r+=1
			# write("./export/"+str(row.category)+"/audio_"+str(i)+str(x)+".wav", sampling_rate, lim_audio)

	print(f"Generated {r} augmented audio data")
	print(f"Total samples: {len(n_list)}")
	new_data = pd.DataFrame(n_list, columns=list(dataset.columns))
	return new_data
		


def audio2mfcc(audio, mfcc_max_length, MFCC_NUM, SAMPLING_RATE):
	'''
	convert audio data to MFCC spectrogram
	'''
	mfcc = librosa.feature.mfcc(audio, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE, hop_length=1024, htk=True)
	if (mfcc_max_length > mfcc.shape[1]):
		pad_width = mfcc_max_length - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
	else:
		mfcc = mfcc[:, :mfcc_max_length]
	
	return mfcc

def get_dataset(path, sampling_rate, mfcc_num, mfcc_max_length, seconds, sample_limit, random_factor, background=True):
	'''
	get X, Y and dataframe of dataset
	'''
	max_length = sampling_rate * seconds
	ds = audio_synthesis(path, sampling_rate, sample_limit, seconds, max_length, random_factor, background)

	x = [audio2mfcc(audio, mfcc_max_length, mfcc_num, sampling_rate) for audio in np.array(ds.audio)]
	x = np.array(x)
	y = np.array(ds.label)
	return x, y, ds


if __name__== "main":
	path = '../input/wav_dataset/'
	sr = 16000
	sample_limit = None
	seconds = 2
	mfcc_num = 20
	mfcc_max_length = 35

	X, Y, df = get_dataset(path, sr, mfcc_num, mfcc_max_length, seconds, sample_limit)


	# import matplotlib.pyplot as plt
	# plt.imshow(X[0])
	# plt.show()
	# plt.savefig('mfcc.png')
