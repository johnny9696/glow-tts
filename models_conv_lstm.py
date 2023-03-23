import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons
import attentions
import monotonic_align

from Speaker_Encoder.speaker_encoder import Convolution_LSTM_cos, LSTM


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = attentions.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = attentions.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self, 
      n_vocab, 
      out_channels, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      n_heads, 
      n_layers, 
      kernel_size, 
      p_dropout, 
      window_size=None,
      block_length=None,
      mean_only=False,
      prenet=False,
      gin_channels=0):

    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.prenet = prenet
    self.gin_channels = gin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if prenet:
      self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      window_size=window_size,
      block_length=block_length,
    )

    self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
    if not mean_only:
      self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)
  
  def forward(self, x, x_lengths, g=None, l=None):

    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    if self.prenet:
      x = self.pre(x, x_mask)
    x = self.encoder(x, x_mask)

    if g is not None and l is None:
      g_exp = g.expand(-1, -1, x.size(2))
      x_dp = torch.cat([torch.detach(x), g_exp], 1)
    elif g is None and l is not None:
      l_exp = l.expand(-1, -1, x.size(2))
      x_dp = torch.cat([torch.detach(x), l_exp], 1)
    elif g is not None and l is not None:
      g_exp = g.expand(-1, -1, x.size(2))
      l_exp = l.expand(-1, -1, x.size(2))
      x_dp = torch.cat([torch.detach(x), g_exp, l_exp], 1)
    else:
      x_dp = torch.detach(x)

    x_m = self.proj_m(x) * x_mask
    if not self.mean_only:
      x_logs = self.proj_s(x) * x_mask
    else:
      x_logs = torch.zeros_like(x_m)
    logw = self.proj_w(x_dp, x_mask)
    return x_m, x_logs, logw, x_mask


class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        attentions.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None,l=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, l=l, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, l=l, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()


class FlowGenerator(nn.Module):
  def __init__(self, 
      n_vocab, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=6,
      p_dropout=0., 
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      n_lang=0,
      gin_channels=0, 
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      lstm_hidden1 = 256,
      lstm_hidden2 = 64,
      lstm_hidden3 =16,
      lstm_l_hidden = 768,
      lstm_num_layers = 3,
      slice_length = 430,
      lstm_kernel = 5,
      speaker_encoder_type=None,
      **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    self.n_speakers = n_speakers
    self.n_lang=n_lang
    self.gin_channels = gin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec
    self.prenet = prenet

    self.lstm_hidden1 = lstm_hidden1
    self.lstm_hidden2 = lstm_hidden2
    self.lstm_hidden3 = lstm_hidden3
    self.lstm_l_hidden = lstm_l_hidden
    self.lstm_num_layers = lstm_num_layers
    self.lstm_kernel = lstm_kernel
    self.slice_length = slice_length

    self.decoder = FlowSpecDecoder(
        out_channels, 
        hidden_channels_dec or hidden_channels, 
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels)
    self.encoder = TextEncoder(
        n_vocab, 
        out_channels, 
        hidden_channels_enc or hidden_channels, 
        filter_channels, 
        filter_channels_dp, 
        n_heads, 
        n_layers_enc, 
        kernel_size, 
        p_dropout, 
        window_size=window_size,
        block_length=block_length,
        mean_only=mean_only,
        prenet=prenet,
        gin_channels=gin_channels)
    emb_size= gin_channels
    if n_speakers >1  and n_lang <1:
        emb_size= gin_channels
    if n_speakers > 1 and n_lang >1 :
        emb_size= gin_channels//2
        self.emb_l = nn.Embedding(self.n_lang, emb_size)
        nn.init.uniform_(self.emb_l.weight, -0.1, 0.1)
    if speaker_encoder_type ==None:
      self.emb_g = nn.Embedding(self.n_speakers, emb_size)
    elif speaker_encoder_type == 'LSTM':
      self.emb_g = LSTM(input_size=self.out_channels, hidden_size=self.lstm_l_hidden,embedding_size=emb_size, num_layers=self.lstm_num_layers)
    elif speaker_encoder_type =='Conv-LSTM':
      self.emb_g=Convolution_LSTM_cos(encoder_dim=self.slice_length, hidden_dim1=self.lstm_hidden1, hidden_dim2=self.lstm_hidden2,hiddem_dim3=self.lstm_hidden3,
        l_hidden=self.lstm_l_hidden, num_layers=self.lstm_num_layers, input_size=self.out_channels,embedding_size=emb_size,kernel=self.lstm_kernel)
    else:
      raise('Wrong Encoder Type')


  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, l=None, gen=False, noise_scale=1., length_scale=1.):
    if g is not None:
      g = self.emb_g(g)
      g = F.normalize(g).unsqueeze(-1)# [b, h]
    if l is not None:
      l=self.emb_l(l)
      l = F.normalize(l).unsqueeze(-1)# [b, h]
    x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g,l=l)

    if gen:
      w = torch.exp(logw) * x_mask * length_scale
      w_ceil = torch.ceil(w)
      y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
      y_max_length = None
    else:
      y_max_length = y.size(2)
    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    if gen:
      attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

      z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
      y, logdet = self.decoder(z, z_mask, g=g, l=l, reverse=True)
      return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
    else:
      z, logdet = self.decoder(y, z_mask, g=g, l=l, reverse=False)
      with torch.no_grad():
        x_s_sq_r = torch.exp(-2 * x_logs)
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
        logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
        logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

        attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
      return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)

  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()
