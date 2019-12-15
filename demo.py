import numpy as np
import torch
import os
import librosa
import torch.nn as nn

import config as cfg
from main import ConvTasNet


if __name__ == '__main__':
    # import parameters
    sample_rate = cfg.sample_rate
    T = cfg.T
    # print('parameters: fft_size:', fft_size, 'hop_size:', hop_size, 'sample_rate:', sample_rate, 'num_mel:', num_mel)

    cPath = os.getcwd()  # current path
    workspace_dir = os.path.join(cPath, 'workspace')
    mix_speech_dir = os.path.join(workspace_dir, 'mix')

    # prepare data
    s1, _ = librosa.load(os.path.join(mix_speech_dir, 'src1.wav'), sr=None)
    s2, _ = librosa.load(os.path.join(mix_speech_dir, 'src2.wav'), sr=None)

    if len(s1) > len(s2):
        s1 = s1[0:len(s2)]
    else:
        s2 = s2[0:len(s1)]

    # normalize
    s1 = (s1 - np.mean(s1))/np.std(s1)
    s2 = (s2 - np.mean(s2))/np.std(s2)

    mix = s1 + s2
    mix_seg = []
    s1_seg = []
    s2_seg = []
    for i in range(len(mix)//T - 1):
        mix_seg.append(torch.from_numpy(mix[i*T:i*T+T]))
        s1_seg.append(s1[i*T:i*T+T])
        s2_seg.append(s2[i*T:i*T+T])
    mix_seg = torch.stack(mix_seg, dim=0)
    print('mix_seg size: ',mix_seg.size())

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N, L, B, H, P, X, R, C, norm_type, causal = 256, 40, 256, 512, 3, 8, 4, 2, "gLN", False
    TasNet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu')
    TasNet.load_state_dict(torch.load('./workspace/Tasnet_params.pkl', map_location=torch.device('cpu')))
    # TasNet.load_state_dict(torch.load('./workspace/Tasnet_params.pkl', map_location=torch.device('cpu')))
    with torch.no_grad():
        out = TasNet(mix_seg)

    out_wave1 = out[:,0,:].reshape(-1).numpy() # 用view会报错, 空间不连续
    out_wave2 = out[:,1,:].reshape(-1).numpy()
    librosa.output.write_wav('./workspace/out1.wav', out_wave1, sr=sample_rate, norm=True)
    librosa.output.write_wav('./workspace/out2.wav', out_wave2, sr=sample_rate, norm=True)
    librosa.output.write_wav('./workspace/mix.wav', mix, sr=sample_rate, norm=True)
