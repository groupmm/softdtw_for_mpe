import os

import librosa
import numpy as np
import pandas as pd

import libdl.data_preprocessing

path_data_basedir = os.path.join("data", 'Schubert_Winterreise')
path_audio_dir = os.path.join(path_data_basedir, '01_RawData', 'audio_wav')
path_annot_dir = os.path.join(path_data_basedir, '02_Annotations', 'ann_audio_note')
path_output = os.path.join(path_data_basedir, 'hcqt_hs512_o6_h5_s1')
os.makedirs(path_output, exist_ok=True)
path_output_annot_p = os.path.join(path_data_basedir, 'pitch_hs512_nooverl')
os.makedirs(path_output_annot_p, exist_ok=True)

audio_list = [os.path.join(root, name)
              for root, dirs, files in os.walk(path_audio_dir)
              for name in files
              if name.endswith(('.mp3', '.wav'))]
num_songs = len(audio_list)

annotation_list = [os.path.join(root, name)
                   for root, dirs, files in os.walk(path_annot_dir)
                   for name in files
                   if name.endswith(('.lab', '.csv', '.txt'))]

# HCQT parameters
fs = 22050
fmin = librosa.note_to_hz('C1')  # MIDI pitch 24
fs_hcqt_target = 50
bins_per_semitone = 3
bins_per_octave = 12 * bins_per_semitone
num_octaves = 6
num_harmonics = 5
num_subharmonics = 1
center_bins = True

for song_id in range(num_songs):

    path_wav = audio_list[song_id]

    song_fn_wav = os.path.basename(path_wav)
    fn_annot = [s for s in annotation_list if song_fn_wav[:-4] in s]

    if not len(fn_annot) == 1:
        print('No single annotation found for ' + song_fn_wav)
    else:
        f_audio, fs_load = librosa.load(path_wav, sr=fs)
        f_hcqt, fs_hcqt, hopsize_hcqt = libdl.data_preprocessing.compute_efficient_hcqt(f_audio, fs=fs, fmin=fmin,
                                                                                        fs_hcqt_target=fs_hcqt_target, bins_per_octave=bins_per_octave, num_octaves=num_octaves,
                                                                                        num_harmonics=num_harmonics, num_subharmonics=num_subharmonics, center_bins=center_bins)
        num_octaves_eff = 6 + np.ceil(np.log2((num_subharmonics + 1)) + np.log2((num_harmonics))).astype(int)

        df = pd.read_csv(fn_annot[0], sep=';', header=0)
        note_events = df.to_numpy()

        f_annot_p = libdl.data_preprocessing.compute_annotation_array_nooverlap(note_events, f_hcqt, fs_hcqt, annot_type='pitch', shorten=1.0)

        np.save(os.path.join(path_output, song_fn_wav[:-4] + '.npy'), f_hcqt)
        np.save(os.path.join(path_output_annot_p, song_fn_wav[:-4] + '.npy'), f_annot_p)

        print('File #' + str(song_id) + ', ' + song_fn_wav[:-4] + ' done.')
