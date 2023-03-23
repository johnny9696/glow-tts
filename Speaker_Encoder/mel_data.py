import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import commons
from utils import load_filepaths_and_text, load_wav_to_torch

from sid_list import sid as sid_list


class Mel_loader(torch.utils.data.Dataset):
    def __init__(self,audio_path, hparams):
        self.hps=hparams
        self.audio_path= load_filepaths_and_text(audio_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = False
        self.add_noise = hparams.add_noise
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audio_path)

    def get_mel_out(self, audio_path):
        audiopath = audio_path[0]
        mel = self.get_mel(audiopath)
        
        return (mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(torch.tensor(melspec), 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        melspec=torch.abs(melspec)/torch.max(torch.abs(melspec))
        return melspec

    def __getitem__(self, index):
        return self.get_mel_out(self.audio_path[index])

    def __len__(self):
        return len(self.audio_path)

class MelCollate():
    """
    batch : [mel_normalize] 2 dimesion shape
    to train auto encoder we need to change in 3 Dimension
    [batch,[frames, mel]] -> [batch, [1,frames, mel]]

    to make all the audio frames same we pad zeros on mel
    than add the channel dimension on the tensor
        """
    def __init__(self, slice_length, hps) -> None:
        
        self.frames = slice_length

    def __call__(self, batch):
        num_mels = batch[0].size(0)
        mel_padded = torch.FloatTensor(len(batch), num_mels, self.frames)
        mel_padded.zero_()
        for i in range(len(batch)):
            mel = batch[i]
            if mel.size(1) > self.frames +100 :
                mel_padded[i, :, :self.frames] = mel[:, 100:self.frames+100]
            elif mel.size(1) > self.frames : 
                mel_padded[i, :, :self.frames] = mel[:,:self.frames]
            else :
                mel_padded[i, :, :mel.size(1)] = mel

        return mel_padded

class MelSID_loader(torch.utils.data.Dataset):
    def __init__(self,audio_path, hparams):
        self.hps=hparams
        self.audio_path= load_filepaths_and_text(audio_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = False
        self.add_noise = hparams.add_noise
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audio_path)

    def get_mel_out(self, audio_path):
        audiopath, sid = audio_path[0], audio_path[1]
        mel = self.get_mel(audiopath)
        sid = self.get_sid(sid)
        return (mel, sid)

    def get_sid(self, sid):
        return torch.tensor(sid_list.index(int(sid)), dtype=torch.long)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(torch.tensor(melspec), 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        melspec=torch.abs(melspec)/torch.max(torch.abs(melspec))
        return melspec 

    def __getitem__(self, index):
        return self.get_mel_out(self.audio_path[index])

    def __len__(self):
        return len(self.audio_path)

class MelSIDCollate():
    """
    batch : [mel_normalize] 2 dimesion shape
    [batch,[mel, frames], sid] -> [batch, [mel, frames], sid]

    to make all the audio frames same we pad zeros on mel
    than add the channel dimension on the tensor
        """
    def __init__(self, slice_length, hps) -> None:
        self.frames = slice_length
        self.speaker= hps.model.output_channel

    def __call__(self, batch):
        num_mels = batch[0][0].size(0)
        mel_padded = torch.FloatTensor(len(batch), num_mels, self.frames)
        mel_padded.zero_()

        sid = torch.LongTensor(len(batch))
        sid = sid.zero_()

        for i in range(len(batch)):
            mel = batch[i][0]
            if mel.size(1) > self.frames :
                mel_padded[i, :, :self.frames] = mel[:, :self.frames]
            else : 
                mel_padded[i, :, :mel.size(1)] = mel
            sid[i] = batch[i][1]
            

        return (mel_padded, sid)


class Mel_GE2E(torch.utils.data.Dataset):
    def __init__(self, audio_path, hps):
        self.audio_path = audio_path
        self.sid_list = os.listdir(self.audio_path)
        """
        if hps.train.utterance != len(sid_list):
            sys.exit('Utterance({}) and Speaker number({}) is not correct'.format(len(sid_list), hps.train.utterance))
        """
        self.speaker_per_wav = []
        for i in self.sid_list:
            tmp = os.path.join(self.audio_path, i)
            wav_list = os.listdir(tmp)
            self.speaker_per_wav.append(len(wav_list))
        self.speaker_min_wav = [x for x in range(min(self.speaker_per_wav))]
        self.max_wav_value = hps.data.max_wav_value
        self.sampling_rate = hps.data.sampling_rate
        self.filter_length = hps.data.filter_length
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self.n_mels = hps.data.n_mel_channels
        self.mel_fmin = hps.data.mel_fmin
        self.mel_fmax = hps.data.mel_fmax
        self.add_noise = hps.data.add_noise
        self.slice_length = hps.data.slice_length
        self.utterance = hps.train.utterance
        self.stft = commons.TacotronSTFT(
            self.filter_length, self.hop_length, self.win_length,
            self.n_mels, self.sampling_rate, self.mel_fmin,
            self.mel_fmax)

        random.seed(1234)
        random.shuffle(self.speaker_min_wav)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            print(filename)
            raise ValueError("{} {} SR doesn't match target {} SR".format(filename, sampling_rate, self.stft.sampling_rate))

        if self.add_noise:
            audio = audio + torch.rand_like(audio)
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(torch.tensor(melspec), 0)

        melspec=torch.abs(melspec)/self.max_wav_value
        return melspec


    def __getitem__(self, index):
        audio_path_tmp= self.sid_list[index]
        wav_path = os.listdir(os.path.join(self.audio_path,audio_path_tmp))
        random.shuffle(wav_path)
        wav_path = wav_path[:self.utterance]
        mel_padded = torch.FloatTensor(self.utterance, self.n_mels, self.slice_length)
        mel_padded.zero_()
        for i, path in enumerate(wav_path):
            mel_db = self.get_mel(os.path.join(self.audio_path,audio_path_tmp,path))
            mel_db=torch.abs(mel_db)/torch.max(torch.abs(mel_db))
            if mel_db.size(1) > self.slice_length :
                mel_padded[i, :, :self.slice_length] = mel_db[:, :self.slice_length]
            else : 
                mel_padded[i, :, :mel_db.size(1)] = mel_db
        return mel_padded


    def __len__(self):
        return len(self.sid_list)
