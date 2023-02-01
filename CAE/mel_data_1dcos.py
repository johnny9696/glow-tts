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
    [batch*2,[mel, frames], sid] -> [batch, [mel, frames], 1 or -1]

    to make all the audio frames same we pad zeros on mel
    than add the channel dimension on the tensor
        """
    def __init__(self, slice_length, hps) -> None:
        self.frames = slice_length

    def __call__(self, batch):
        num_mels = batch[0][0].size(0)
        mel_padded_A = torch.FloatTensor(len(batch)//2, num_mels, self.frames)
        mel_padded_A.zero_()
        mel_padded_B = torch.FloatTensor(len(batch)//2, num_mels, self.frames)
        mel_padded_B.zero_()

        sid_A = torch.LongTensor(len(batch)//2)
        sid_A = sid_A.zero_()
        sid_B = torch.LongTensor(len(batch)//2)
        sid_B = sid_B.zero_()

        NorF = torch.LongTensor(len(batch)//2)
        NorF = NorF.zero_()

        for i in range(len(batch)//2):
            mel_A = batch[i][0]
            mel_B = batch[len(batch)//2 +i][0]
            if mel_A.size(1) > self.frames :
                mel_padded_A[i, :, :self.frames] = mel_A[:, :self.frames]
            else : 
                mel_padded_A[i, :, :mel_A.size(1)] = mel_A
            if mel_B.size(1) > self.frames :
                mel_padded_B[i, :, :self.frames] = mel_B[:, :self.frames]
            else : 
                mel_padded_B[i, :, :mel_B.size(1)] = mel_B
            sid_A[i] = batch[i][1]
            sid_B[i] = batch[len(batch)//2+i][1]

            if sid_A[i] == sid_B[i] :
                NorF[i] = torch.tensor(1)
            else :
                NorF[i] = torch.tensor(-1)
            
        return (mel_padded_A,mel_padded_B, sid_A, sid_B, NorF)