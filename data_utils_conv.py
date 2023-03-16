import random
import numpy as np
import torch
import torch.utils.data

import commons 
from utils import load_wav_to_torch, load_filepaths_and_text, load_flac_to_torch
from text import IPA_id, text_id
from text.multi_apha import letter_
from text.lang_type import l2num as lt
from librosa.feature import melspectrogram
import librosa

"""Multi speaker & Lingual version"""
class TextMelSpeakerLangLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False) # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        if getattr(hparams, "cmudict_path", None) is not None:
          self.cmudict = cmudict.CMUDict(hparams.cmudict_path)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.feature_vector = hparams.feature_files
        with open(self.feature_vector,'r',encoding='UTF-8') as f:
            d_w  = f.read().split('\n')
        self.feature_dict = dict()
        for i in d_w:
            a_sid = i.split('|')
            if len(a_sid)>=2:
                self.feature_dict[a_sid[1]] = a_sid[0]
        #print(self.feature_dict)

        #self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
      audiopaths_sid_text_new = []
      for audiopath, sid,lang, text in self.audiopaths_sid_text:
        if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
          audiopaths_sid_text_new.append([audiopath, sid, lang, text])
      self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, lang, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3]
        lang, e_lang=self.get_lang(lang)
        text = self.get_text(text,e_lang)
        mel = self.get_mel(audiopath)
        sid, feature_mel = self.get_sid(sid)
        
        return (text, mel, sid,lang, feature_mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            if filename[-4:]=='flac':
                audio,sampling_rate=load_flac_to_torch(filename)
            else:
                audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text,lang):
        text_norm = IPA_id.text_to_sequence(text,lang)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(letter_)) # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        a_path = self.feature_dict[sid]
        feature_mel = self.get_mel(a_path)
        sid = torch.IntTensor([int(sid)])
        return sid, feature_mel
    
    def get_lang(self, lang):
        lang,e_lang=lt(lang)
        lang=torch.IntTensor([int(lang)])
        return lang,e_lang

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerLangCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, slice_length=430):
        self.n_frames_per_step = n_frames_per_step
        self.slice_length = slice_length

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, sid, lang, feature_mel]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        mel_emb = torch.FloatTensor(len(batch), num_mels, self.slice_length)
        mel_emb.zero_()

        sid = torch.LongTensor(len(batch))
        lang = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            f_mel = batch[ids_sorted_decreasing[i]][4]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            mel_norm = torch.abs(f_mel)/torch.max(torch.abs(f_mel))
            if mel_norm.size(1) > self.slice_length +100:
                mel_emb[i, :, :self.slice_length] = mel_norm[:, 100:self.slice_length+100]
            elif mel_norm.size(1) > self.slice_length :
                mel_emb[i, :, :self.slice_length] = mel_norm[:, :self.slice_length]
            else :
                mel_emb[i, :, :mel_norm.size(1)] = mel_norm

            sid[i] = batch[ids_sorted_decreasing[i]][2]
            lang[i] = batch[ids_sorted_decreasing[i]][3]

        return text_padded, input_lengths, mel_padded, output_lengths, sid, mel_emb, lang
