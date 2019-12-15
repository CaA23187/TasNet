import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import librosa

cPath = os.getcwd()  # current path
workspace_dir = os.path.join(cPath, 'workspace')
hdf5_dir = os.path.join(workspace_dir, 'spectra.hdf5')

with h5py.File(hdf5_dir, 'r') as hf:
    src_1 = hf['test']['src_1'][:]
    src_2 = hf['test']['src_2'][:]
    mix = hf['test']['mix'][:]

# cPath = os.getcwd()  # current path
# workspace_dir = os.path.join(cPath, 'workspace')
# mix_speech_dir = os.path.join(workspace_dir, 'mix')
#
# # prepare data
# s1, _ = librosa.load(os.path.join(mix_speech_dir, 'src1.wav'), sr=None)
# s2, _ = librosa.load(os.path.join(mix_speech_dir, 'src2.wav'), sr=None)
#
# s1 = (s1 - np.mean(s1))/np.std(s1)
# s2 = (s2 - np.mean(s2))/np.std(s2)
# print(np.mean(s1), np.std(s1))
#
# # s1 = (s1 - np.min(s1))/(np.max(s1)-np.mean(s1))
# # s2 = (s2 - np.min(s2))/(np.max(s2)-np.mean(s2))
#
# # s1 = s1 / np.sqrt(sum(s1 ** 2))
# # s2 = s2 / np.sqrt(sum(s2 ** 2))
#
# if len(s1) > len(s2):
#     s1 = s1[0:len(s2)]
# else:
#     s2 = s2[0:len(s1)]
#
# mix = s1+s2
#
# plt.figure()
# plt.plot(s1)
# plt.figure()
# plt.plot(s2)
# plt.figure()
# plt.plot(mix)
# plt.show()

