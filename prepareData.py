import librosa
import h5py
import numpy as np
import re
import os
import random
import shutil
import time
import argparse

import config as cfg


# print('h5py version: ', h5py.__version__)
def real_to_complex(pd_abs_x, gt_x):
    """Recover pred spectrogram's phase from ground truth's phase. 
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)
      
    Returns:
      2d complex array, (n_time, n_freq)
    """
    theta = np.angle(gt_x)
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx

def divideData(args):
    divide_factor = args.divide_factor
    data_dir = args.data_dir

    cPath = os.getcwd()  # current path
    workspace_dir = os.path.join(cPath, 'workspace')
    t1 = time.time()

    # create dir
    speech_dir = os.path.join(workspace_dir, 'speechFile')
    if not os.path.exists(os.path.join(speech_dir, 'train_speech')):
        os.makedirs(os.path.join(speech_dir, 'train_speech'))
    if not os.path.exists(os.path.join(speech_dir, 'test_speech')):
        os.makedirs(os.path.join(speech_dir, 'test_speech'))

    # random pick train_speech and test_speech
    name_list = os.listdir(data_dir)
    random.shuffle(name_list)
    sep = int(divide_factor * len(name_list))
    for name in name_list[:sep]:
        origin_path = os.path.join(data_dir, name)
        new_file_name = os.path.join(speech_dir, 'train_speech', name)
        shutil.move(origin_path, new_file_name)
    for name in name_list[sep:]:
        origin_path = os.path.join(data_dir, name)
        new_file_name = os.path.join(speech_dir, 'test_speech', name)
        shutil.move(origin_path, new_file_name)
    print("Divide data successfully, ", 100 * divide_factor, '% to train. Using time:', time.time() - t1)


def melspectrogram(spec, sr=16000, n_fft=1024, n_mels=128, power=2.0):
    '''
    Compute a mel-scaled spectrogram through a spectrogram
    Args:
        spec: input spectrogram
        sr: sample rate
        n_fft: number of FFT components
        n_mels: number of Mel bands to generate
        power: Exponent for the magnitude melspectrogram
    Out:
        mel-scaled spectrogram (n_mels, t)
    '''
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    return np.dot(mel_basis, np.abs(spec) ** power)  # (n_mels, t) = (n_mels, FFT) * (FFT, t)


def gen_mix_list(path, epoch):
    '''
    generate src 1 and src 2 name to mix up in the future.
    Args:
        path: the director of wav file
        epoch: Number of loop generations
    return:
        mix_list: the list of (src_1 name, src_2 name)
    '''
    file_list = os.listdir(path)
    mix_list = []
    for i in range(epoch):
        file_list2 = file_list
        random.shuffle(file_list)
        file_list = os.listdir(path)
        for s1, s2 in zip(file_list, file_list2):
            # 找到数字
            id1 = re.findall(r'\d+', s1)
            id2 = re.findall(r'\d+', s2)

            if int(id1[-1]) == int(id2[-1]):
                continue
            mix_list.append((s1, s2))
    return mix_list


def uniform_time_len(spec_list):
    time_bin = []
    for s in spec_list:
        time_bin.append(s.shape[1])
    mean_time = int(np.mean(time_bin))  # 求平均时间帧
    spec_withsameDimList = []  # 统一时间帧为平均
    for spec in spec_list:
        if spec.shape[1] > mean_time:
            spec_withsameDimList.append(spec[:, :mean_time])
        else:
            spec_withsameDimList.append(
                np.concatenate((spec, np.zeros([spec.shape[0], mean_time - spec.shape[1]])), axis=1))
    return spec_withsameDimList


def write_spectra_to_HDF5(audio_dir, hdf5_path, name_list, data_type, sample_rate=16000, T=96):
    '''
    Write features to hdf5 file. Features include log-mel_feature, mixture spectrogram,
    src 1 magnitude spec and src 2 magnitude spec.

    Args: audio_dir: (str) path of audio file(.wav)
          hdf5_path: (str) path of hdf5 file
          name_list: ((str,str)) list of src_1 and src_2 name
          data_type: (str) type of dataset, train or test
          T: (int) segment length of wav
    '''
    assert data_type in ['train', 'test']

    mix_seg = []
    y1_seg = []
    y2_seg = []

    for s1, s2 in name_list:
        s1_path = os.path.join(audio_dir, s1)
        s2_path = os.path.join(audio_dir, s2)
        y1, _ = librosa.load(s1_path, sr=None)
        y2, _ = librosa.load(s2_path, sr=None)

        # uniform the length
        if len(y1)>len(y2):
            y1 = y1[:len(y2)]
        else:
            y2 = y2[:len(y1)]

        # normalize
        y1 = (y1 - np.mean(y1))/np.std(y1)
        y2 = (y2 - np.mean(y2))/np.std(y2)

        mix = y1 + y2

        for i in range(len(mix)//T - 1):
            mix_seg.append(mix[i*T:i*T+T])
            y1_seg.append(y1[i*T:i*T+T])
            y2_seg.append(y2[i*T:i*T+T])

    # write to hdf5 file
    with h5py.File(hdf5_path, 'a') as hf:
        if data_type not in hf.keys():
            hf.create_group(data_type)

        hf[data_type].create_dataset(
            name='src_1',
            data=y1_seg,
            dtype=np.float32)
        hf[data_type].create_dataset(
            name='src_2',
            data=y2_seg,
            dtype=np.float32)
        hf[data_type].create_dataset(
            name='mix',
            data=mix_seg,
            dtype=np.float32)
    print('write ', data_type, ' successfully')


def mix_data(args):
    mix_epoch = args.mix_epoch

    cPath = os.getcwd()  # 当前路径
    train_dir = os.path.join(cPath, 'workspace', 'speechFile', 'train_speech')
    test_dir = os.path.join(cPath, 'workspace', 'speechFile', 'test_speech')
    t1 = time.time()

    # parameters init
    sample_rate = cfg.sample_rate
    T = cfg.T
    # hop_size = cfg.hop_size
    # num_mel = cfg.num_mel

    # get mix file name
    mix_tr_list = gen_mix_list(train_dir, mix_epoch)
    mix_te_list = gen_mix_list(test_dir, mix_epoch)

    # Create hdf5 file
    hdf5_path = os.path.join(cPath, 'workspace', 'spectra.hdf5')
    with h5py.File(hdf5_path, 'w') as hf:
        hf.attrs['sample_rate'] = sample_rate
        hf.attrs['T'] = T

    # Write out features to hdf5
    write_spectra_to_HDF5(train_dir, hdf5_path, mix_tr_list, 'train', sample_rate, T)
    write_spectra_to_HDF5(test_dir, hdf5_path, mix_te_list, 'test', sample_rate, T)

    print('Generated ', len(mix_tr_list), 'mixture train speech, ', len(mix_te_list), 'mixture test speech, ')
    print("Write out to hdf5_path: %s" % hdf5_path, " using time: ", time.time() - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_features = subparsers.add_parser('divide_data')
    parser_calculate_features.add_argument('--data_dir', type=str, required=True)
    parser_calculate_features.add_argument('--divide_factor', type=float, required=True)

    parser_calculate_features = subparsers.add_parser('mix_data')
    parser_calculate_features.add_argument('--mix_epoch', type=int, required=True)

    args = parser.parse_args()
    if args.mode == 'divide_data':
        divideData(args)
    elif args.mode == 'mix_data':
        mix_data(args)
    else:
        raise Exception("Error!")
