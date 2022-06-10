#!/usr/bin/env python
# coding: utf-8

# In[403]:


get_ipython().system('pip install pydub')


# In[449]:


import pydub
import os 
import shutil
from pydub import AudioSegment
import librosa
import numpy as np

#make a directory of all the .lab files

def get_chord_files(chord_lab_path):
    cwd = os.getcwd()
    chord_dir = os.path.join(cwd,'chord_files')
    if not os.path.isdir(chord_dir):
        os.makedirs(chord_dir)
    for root, dirs, files in os.walk(chord_lab_path):
        for file in files:
            if file.endswith(".lab"):
                shutil.copyfile(os.path.join(root,file), os.path.join(chord_dir,file))
    return 

def convert_to_wav(songs_path):
    cwd = os.getcwd()
    wav_dir = os.path.join(cwd,'songs_wav')
    if not os.path.isdir(wav_dir):
        os.makedirs(wav_dir)
    for root, dirs, files in os.walk(songs_path):
        for file in files:
            if file.endswith('.mp3'):
                track = AudioSegment.from_file(os.path.join(root,file),format='mp3')
                waveform = track.export(os.path.join(cwd,wav_dir,file[:-3]+'wav'),format = 'wav')
                print('exported one wav file')
            elif file.endswith('.m4a'):
                track = AudioSegment.from_file(os.path.join(root,file),format='m4a')
                waveform = track.export(os.path.join(cwd,wav_dir,file[:-3]+'wav'),format = 'wav')
                print('exported one wav file')
    return

def make_stft_dataset(songs_path):
    cwd = os.getcwd()
    stft_dir = os.path.join(cwd,'stft_dataset')
    if not os.path.isdir(stft_dir):
        os.makedirs(stft_dir)
    for file in os.listdir(songs_path):
        waveform,sr = librosa.load(os.path.join(songs_path,file))
        stft = librosa.stft(waveform)
        stft_mag = np.abs(stft)
        np.save(os.path.join(stft_dir,file[:-4]),stft_mag)
        print("Saved {file_name} stft".format(file_name = file[:-4]))
    return 


# In[432]:


get_chord_files('C:/Users/admitos/Desktop/chord_files')
#convert_to_wav('C:/Users/admitos/Desktop/A Hard Day Night DELUXE')
make_stft_dataset('C:/Users/admitos/songs_wav')
#make_stft_dataset('C:/Users/admitos/Desktop/Yellow Submarine DELUXE')


# In[568]:


import pandas as pd
import os
import numpy as np
import pickle
import random

SAMPLING_RATE = 22050 #in hertz
N_FFT = 2048
WIN_LENGTH = 2048 
HOP_LENGTH = 2048 // 4


def get_all_chords(chord_dir):
    unique_chords = set()
    for file in os.listdir(chord_dir):
        path = os.path.join(chord_dir,file)
        contents = pd.read_csv(path,delim_whitespace=True)
        chords = contents.iloc[:,2].values

        for chord in chords: 
            unique_chords.add(chord)
    return unique_chords

def assign_index_to_chords(chords):
    mapping = {}
    for i,c in enumerate(chords):
        mapping[c] = i
        with open('chord_idxs.pkl', 'wb') as f:
            pickle.dump(mapping, f)
    #print("Mapping is: ", mapping)
    return 

def make_frame_labels(labels_path,chords_set):
    all_labels = []
    for file in os.listdir(labels_path):
        path = os.path.join(labels_path,file)
        contents = pd.read_csv(path,delim_whitespace=True).values
        finish_time = contents[-1,1]
        time_frames = librosa.time_to_frames(finish_time)
        onset_labels = np.zeros((len(chords_set),time_frames))
        for content in contents:
            start_frame = librosa.time_to_frames(content[0])
            end_frame = librosa.time_to_frames(content[1])
            onset_labels[chords_set[content[2]],start_frame:end_frame+1] = 1
        all_labels.append(onset_labels)
    return all_labels

def make_stft_dataset_2(songs_path,sampling_rate,n_fft,hop_length,win_length):
    cwd = os.getcwd()
    print(cwd)
    stft_dir = os.path.join(cwd,'stft_dataset')
    print(stft_dir)
    print(songs_path)
    if not os.path.isdir(stft_dir):
        os.makedirs(stft_dir)
    for file in os.listdir(songs_path):
        waveform,sr = librosa.load(os.path.join(songs_path,file),sr=sampling_rate)
        stft = librosa.stft(waveform,n_fft=n_fft,hop_length=hop_length,win_length=win_length)
        stft_mag = np.abs(stft)
        np.save(os.path.join(stft_dir,file[:-4]),stft_mag)
        print("Saved {file_name} stft".format(file_name = file[:-4]))
    return 

if __name__ == '__main__':
    cwd = os.getcwd()
    #for file in os.listdir('songs_wav'):
        #waveform,sr = librosa.load(os.path.join(cwd,'songs_wav',file))
        #stft = librosa.stft(waveform,)
        #print('no error with wav file')
    # assign_index_to_chords(get_all_chords('chord_files'))
    # with open('chord_idxs.pkl', 'rb') as f:
    #   loaded_dict = pickle.load(f)
    temp = os.path.join(cwd,'songs_wav')
    song_path = os.path.join(temp,'A Hard Day_s Night.npy')
    #make_stft_dataset_2(temp,SAMPLING_RATE,N_FFT,HOP_LENGTH,WIN_LENGTH)
    
    """or f in os.listdir('stft_dataset'):
        stft = np.load(os.path.join(cwd,'stft_dataset',f))
        print(stft.shape)

    assign_index_to_chords(get_all_chords('one_label'))
    with open('chord_idxs.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    print(loaded_dict)

    frame_list = make_frame_labels('one_label',loaded_dict)
    print(frame_list[0].shape) """


# In[635]:


def make_onset_labels(labels_path,chords_set):
    all_labels = []
    for file in os.listdir(labels_path):
        path = os.path.join(labels_path,file)
        contents = pd.read_csv(path,delim_whitespace=True)
        print(contents.shape)
        finish_time = contents.values[-1,1]
        time_frames = librosa.time_to_frames(finish_time)
        #print(contents.iloc[:,[0,2]].values)
        contents = contents.iloc[:,[0,2]].values
        onset_labels = np.zeros((len(chords_set),time_frames))
        for content in contents:
            print("content is:", content[0], content[1])
            print("coordinates are: ", chords_set[content[1]], librosa.time_to_frames(content[0]))            
            onset_labels[chords_set[content[1]],librosa.time_to_frames(content[0])] = 1
            #print(onset_labels)
        all_labels.append(onset_labels)
    return all_labels

def get_sample_from_song(sec,song, chords):
    time_frames = librosa.time_to_frames(sec)
    print('time frames corresponding to sec',time_frames)
    start = random.randint(0,song.shape[1]-time_frames)
    print('print starting frame to sample ',start)
    sample_x = song[:,start:start+time_frames]
    sample_y = chords[:,start:start+time_frames]
    return (sample_x,sample_y)


# In[650]:


print('song dimensions are: ', stft.shape)
print('chords dimensions are: ', onset_list[0].shape)
dataloader_x = []
dataloader_y = []
for i in range(10):
    x,y = get_sample_from_song(10,stft,onset_list[0])
    dataloader_x.append(x)
    dataloader_y.append(y)
tensor_x = np.array(dataloader_x)
tensor_y = np.array(dataloader_y)
print(tensor_x.shape,tensor_y.shape) 


# In[658]:


from numpy import savetxt
print(type(tensor_x))
arr_reshaped = tensor_x.reshape(tensor_x.shape[0]*tensor_x.shape[1], -1)
print(arr_reshaped.shape)
savetxt('data.csv', arr_reshaped, delimiter=',')


# In[648]:


q=0
for f in os.listdir('stft_dataset'):
    stft = np.load(os.path.join(cwd,'stft_dataset',f))
    print(stft.shape)
    if (q == 3):
        break
    q = q + 1

all_chords_found = get_all_chords('C:/Users/admitos/Desktop/chord_files')
print(len(all_chords_found))
assign_index_to_chords(all_chords_found)
with open('chord_idxs.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    #print(loaded_dict)

onset_list = make_onset_labels('C:/Users/admitos/Desktop/chord_files',loaded_dict)
print(len(onset_list))

#frame_list = make_frame_labels('one_label',loaded_dict)
#print(frame_list[0].shape)


# In[662]:


import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
f = open('chord_idxs.pkl', 'rb')
dict = pickle.load(f)


# In[663]:


entries_chords = os.listdir('C:/Users/admitos/Desktop/chord_files')
entries_songs_1 = os.listdir('C:/Users/admitos/Desktop/A Hard Day Night DELUXE')
entries_songs_2 = os.listdir('C:/Users/admitos/Desktop/Abbey Road DELUXE')
entries_songs_3 = os.listdir('C:/Users/admitos/Desktop/Beatles For Sale DELUXE')
entries_songs_4 = os.listdir('C:/Users/admitos/Desktop/Before The Beatles DELUXE')
entries_songs_5 = os.listdir('C:/Users/admitos/Desktop/Help! DELUXE')
entries_songs_6 = os.listdir('C:/Users/admitos/Desktop/Let It Be DELUXE')
entries_songs_7 = os.listdir('C:/Users/admitos/Desktop/Please Please Me DELUXE')
entries_songs_8 = os.listdir('C:/Users/admitos/Desktop/Magical Mystery Tour DELUXE')
entries_songs_9 = os.listdir('C:/Users/admitos/Desktop/Revolver DELUXE')
entries_songs_10 = os.listdir('C:/Users/admitos/Desktop/Rubber Soul DELUXE')
entries_songs_11 = os.listdir('C:/Users/admitos/Desktop/Sgt. Pepper_s Lonely Hearts Club Band DELUXE')
entries_songs_12 = os.listdir('C:/Users/admitos/Desktop/The White Album DELUXE')
entries_songs_13 = os.listdir('C:/Users/admitos/Desktop/With The Beatles DELUXE')
entries_songs_14 = os.listdir('C:/Users/admitos/Desktop/Yellow Submarine DELUXE')

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def Convert(string):
    list1=[]
    list1[:0]=string
    return list1


# In[664]:


random_string = 'mdadmsna.mp3'
max_length = max(len(entries_songs_1),len(entries_songs_2),len(entries_songs_3),len(entries_songs_4),len(entries_songs_5),
                 len(entries_songs_6),len(entries_songs_7),len(entries_songs_8),len(entries_songs_9),len(entries_songs_10),
                 len(entries_songs_11),len(entries_songs_12),len(entries_songs_13),len(entries_songs_14))
print(max_length)
entries_list = [entries_songs_1, entries_songs_2, entries_songs_3, entries_songs_4, entries_songs_5, entries_songs_6, entries_songs_7,
                entries_songs_8, entries_songs_9, entries_songs_10, entries_songs_11, entries_songs_12, entries_songs_13, entries_songs_14]

for e in entries_list:
    if (e==entries_songs_6):
        continue
    e.extend([random_string] * (len(entries_songs_6)-len(e)))


# In[665]:


my_list = []
for i in entries_chords:
    s = Convert(i)
    my_list.append(s)

for i in my_list:
    del i[0:5]
    del i[-4:]


# In[666]:


list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12,list13,list14 = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
for a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 in zip(entries_songs_1, entries_songs_2, entries_songs_3, entries_songs_4, entries_songs_5,
                    entries_songs_6,entries_songs_7, entries_songs_8, entries_songs_9, entries_songs_10, entries_songs_11, 
                    entries_songs_12, entries_songs_13, entries_songs_14):
    s1,s8 = Convert(a1), Convert(a8)
    s2,s9 = Convert(a2), Convert(a9)
    s3,s10 = Convert(a3), Convert(a10)
    s4,s11 = Convert(a4), Convert(a11)
    s5,s12 = Convert(a5), Convert(a12)
    s6,s13 = Convert(a6), Convert(a13)
    s7,s14 = Convert(a7), Convert(a14)
    list1.append(s1)
    list2.append(s2)
    list3.append(s3)
    list4.append(s4)
    list5.append(s5)
    list6.append(s6)
    list7.append(s7)
    list8.append(s8)
    list9.append(s9)
    list10.append(s10)
    list11.append(s11)
    list12.append(s12)
    list13.append(s13)
    list14.append(s14)


# In[667]:


for a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 in zip(list1, list2, list3, list4, list5, list6, list7, list8, list9, 
                                                  list10, list11, list12, list12, list14):
    print(type(a1))
    print(type(a2))
    print(type(a3))
    print(type(a4))
    print(type(a5))
    print(type(a6))
    print(type(a7))
    print(type(a8))
    print(type(a9))
    print(type(a10))
    print(type(a11))
    print(type(a12))
    print(type(a13))
    print(type(a14))


# In[668]:


for a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 in zip(list1, list2, list3, list4, list5, list6, list7, list8, list9, 
                                                  list10, list11, list12, list13, list14):
    del a1[-4:]
    del a2[-4:]
    del a3[-4:]
    del a4[-4:]
    del a5[-4:]
    del a6[-4:]
    del a7[-4:]
    del a8[-4:]
    del a9[-4:]
    del a10[-4:]
    del a11[-4:]
    del a12[-4:]
    del a13[-4:]
    del a14[-4:]


# In[669]:


for t,q in enumerate(my_list):
    temp = ''.join(q)
    entries_chords[t] = temp
    
Lists = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, list11, list12, list13, list14] 
for l,entry in zip(Lists, entries_list):
    print(type(l),type(entry))
    for t,q in enumerate(l):
        temp = ''.join(q)
        entry[t] = temp


# In[670]:


for t in entries_songs_14:
    print(t)


# In[699]:


final_songs = []
for l,i in enumerate(entries_chords):
    rate_max = 0
    rate_max_1 = 0
    rate_max_2 = 0
    rate_max_3 = 0
    rate_max_4 = 0
    rate_max_5 = 0
    rate_max_6 = 0
    rate_max_7 = 0
    rate_max_8 = 0
    rate_max_9 = 0
    rate_max_10 = 0
    rate_max_11 = 0
    rate_max_12 = 0
    rate_max_13 = 0
    rate_max_14 = 0
    print("The chords name is: ", entries_chords[l])    
    for k,j in enumerate(entries_songs_1):
        rate = similar(i,j)
        if (rate>rate_max_1):
            rate_max_1 = rate
            pos_max_1 = k
    
    for k,j in enumerate(entries_songs_2):
        rate = similar(i,j)
        if (rate>rate_max_2):
            rate_max_2 = rate
            pos_max_2 = k
            
    for k,j in enumerate(entries_songs_3):
        rate = similar(i,j)
        if (rate>rate_max_3):
            rate_max_3 = rate
            pos_max_3 = k
            
    for k,j in enumerate(entries_songs_4):
        rate = similar(i,j)
        if (rate>rate_max_4):
            rate_max_4 = rate
            pos_max_4 = k

    for k,j in enumerate(entries_songs_5):
        rate = similar(i,j)
        if (rate>rate_max_5):
            rate_max_5 = rate
            pos_max_5 = k

    for k,j in enumerate(entries_songs_6):
        rate = similar(i,j)
        if (rate>rate_max_6):
            rate_max_6 = rate
            pos_max_6 = k

    for k,j in enumerate(entries_songs_7):
        rate = similar(i,j)
        if (rate>rate_max_7):
            rate_max_7 = rate
            pos_max_7 = k
            
    for k,j in enumerate(entries_songs_8):
        rate = similar(i,j)
        if (rate>rate_max_8):
            rate_max_8 = rate
            pos_max_8 = k
            
    for k,j in enumerate(entries_songs_9):
        rate = similar(i,j)
        if (rate>rate_max_9):
            rate_max_9 = rate
            pos_max_9 = k
            
    for k,j in enumerate(entries_songs_10):
        rate = similar(i,j)
        if (rate>rate_max_10):
            rate_max_10 = rate
            pos_max_10 = k

    for k,j in enumerate(entries_songs_11):
        rate = similar(i,j)
        if (rate>rate_max_11):
            rate_max_11 = rate
            pos_max_11 = k
            
    for k,j in enumerate(entries_songs_12):
        rate = similar(i,j)
        if (rate>rate_max_12):
            rate_max_12 = rate
            pos_max_12 = k 
            
    for k,j in enumerate(entries_songs_13):
        rate = similar(i,j)
        if (rate>rate_max_13):
            rate_max_13 = rate
            pos_max_13 = k   
            
    for k,j in enumerate(entries_songs_14):
        rate = similar(i,j)
        if (rate>rate_max_14):
            rate_max_14 = rate
            pos_max_14 = k
            
    rate_max = max(rate_max_1, rate_max_2, rate_max_3, rate_max_4, rate_max_5, rate_max_6, rate_max_7, rate_max_8,
                  rate_max_9, rate_max_10, rate_max_11, rate_max_12, rate_max_13, rate_max_14)
    
    if (rate_max == rate_max_1):
        pos_max = pos_max_1
        print("The most similar song title is: ", entries_songs_1[pos_max])
        print("The position in the list is: ",  pos_max)
        final_songs.append(entries_songs_1[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/A Hard Day Night DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/A Hard Day Night DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/A Hard Day Night DELUXE' + '/' +  entries_songs_1[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    elif (rate_max == rate_max_2):
        pos_max = pos_max_2
        print("The most similar song title is: ", entries_songs_2[pos_max])
        print("The position in the song list is: ",  pos_max)
        final_songs.append(entries_songs_2[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Abbey Road DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Abbey Road DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Abbey Road DELUXE' + '/' +  entries_songs_2[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    elif (rate_max == rate_max_3):
        pos_max = pos_max_3
        print("The most similar song title is: ", entries_songs_3[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_3[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Beatles For Sale DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Beatles For Sale DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Beatles For Sale DELUXE' + '/' +  entries_songs_3[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    elif (rate_max == rate_max_4):
        pos_max = pos_max_4
        print("The most similar song title is: ", entries_songs_4[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_4[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Before The Beatles DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Before The Beatles DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Before The Beatles DELUXE' + '/' +  entries_songs_4[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    elif (rate_max == rate_max_5):
        pos_max = pos_max_5
        print("The most similar song title is: ", entries_songs_5[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_5[pos_max])  
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Help! DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Help! DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Help! DELUXE' + '/' +  entries_songs_5[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1
                
    elif (rate_max == rate_max_6):
        pos_max = pos_max_6
        print("The most similar song title is: ", entries_songs_6[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_6[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Let It Be DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Let It Be DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Let It Be DELUXE' + '/' +  entries_songs_6[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1
                
    elif (rate_max == rate_max_7):
        pos_max = pos_max_7
        print("The most similar song title is: ", entries_songs_7[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_7[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Please Please Me DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Please Please Me DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Please Please Me DELUXE' + '/' +  entries_songs_7[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1 
                
    elif (rate_max == rate_max_8):
        pos_max = pos_max_8
        print("The most similar song title is: ", entries_songs_8[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_8[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Magical Mystery Tour DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Magical Mystery Tour DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Magical Mystery Tour DELUXE' + '/' +  entries_songs_8[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1 
                
    elif (rate_max == rate_max_9):
        pos_max = pos_max_9
        print("The most similar song title is: ", entries_songs_9[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_9[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Revolver DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Revolver DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Revolver DELUXE' + '/' +  entries_songs_9[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1 
                
    elif (rate_max == rate_max_10):
        pos_max = pos_max_10
        print("The most similar song title is: ", entries_songs_10[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_10[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Rubber Soul DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Rubber Soul DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Rubber Soul DELUXE' + '/' +  entries_songs_10[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1 
        
    elif (rate_max == rate_max_11):
        pos_max = pos_max_11
        print("The most similar song title is: ", entries_songs_11[pos_max])
        print("The position in the song list is: ",  pos_max)                
        final_songs.append(entries_songs_11[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Sgt. Pepper_s Lonely Hearts Club Band DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Sgt. Pepper_s Lonely Hearts Club Band DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Sgt. Pepper_s Lonely Hearts Club Band DELUXE' + '/' +  entries_songs_11[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1 
                
    elif (rate_max == rate_max_12):
        pos_max = pos_max_12
        print("The most similar song title is: ", entries_songs_12[pos_max])
        print("The position in the song list is: ",  pos_max)      
        final_songs.append(entries_songs_12[pos_max]) 
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/The White Album DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/The White Album DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/The White Album DELUXE' + '/' +  entries_songs_12[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    elif (rate_max == rate_max_13):
        pos_max = pos_max_13
        print("The most similar song title is: ", entries_songs_13[pos_max])
        print("The position in the song list is: ",  pos_max)        
        final_songs.append(entries_songs_13[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/With The Beatles DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/With The Beatles DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/With The Beatles DELUXE' + '/' +  entries_songs_13[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1

    else:
        pos_max = pos_max_14
        print("The most similar song title is: ", entries_songs_14[pos_max])
        print("The position in the sogn list is: ",  pos_max)        
        final_songs.append(entries_songs_14[pos_max])
        t = 0
        for file in os.listdir('C:/Users/admitos/Desktop/Yellow Submarine DELUXE'):
            if (t!=pos_max):
                t = t + 1
                continue
            else:
                source = 'C:/Users/admitos/Desktop/Yellow Submarine DELUXE' + '/' + file
                print(source)
                dest = 'C:/Users/admitos/Desktop/Yellow Submarine DELUXE' + '/' +  entries_songs_14[t] + '.wav'
                print(dest)
                os.rename(source, dest)
                t = t + 1
    
    print("Τhe maximum similarity rate is: ", rate_max)
    print("")
    


# In[672]:


q = 1
for i,j in zip(entries_chords,final_songs):
    print(q,":", i)
    print("   ",j)
    print("")
    q = q + 1


# In[690]:


k = 0
for file in os.listdir('C:/Users/admitos/Desktop/chord_files'):
    source = 'C:/Users/admitos/Desktop/chord_files' + '/' + file
    print(source)
    dest = 'C:/Users/admitos/Desktop/chord_files' + '/' +  entries_chords[k] + '.lab'
    print(dest)
    os.rename(source, dest)
    k = k + 1


# In[701]:


k = 0
for file in os.listdir('C:/Users/admitos/Desktop/final_songs'):
    k = k + 1
    
print(k)


# In[ ]:




