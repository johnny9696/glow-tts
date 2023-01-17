import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


from data_utils import TextMelLoader, TextMelCollate , TextMelSpeakerLoader, TextMelSpeakerCollate
import models
from CAE.CAE import Convolution_Auto_Encoder as CAE
import commons
import utils
from text.symbols import symbols

import audio_processing as ap
import librosa
                            

global_step = 2


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.current_device()
  print(n_gpus)
  rank=0

  hps = utils.get_hparams()
  train_and_eval(rank,0,hps)
  
  

def train_and_eval(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))


  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(n_gpus)


  train_dataset = TextMelSpeakerLoader(hps.data.training_files, hps.data)
  collate_fn = TextMelSpeakerCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)
  if rank == 0:
    val_dataset = TextMelSpeakerLoader(hps.data.validation_files, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=1, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

  AE_model = CAE(encoder_dim=1, hidden_1dim=3,
     kernel=5)

    #load model dict
  checkpoint_path = "/media/caijb/data_drive/autoencoder/log/kernel5"
  checkpoint_path = utils.latest_checkpoint_path(checkpoint_path)
  AE_model, _, _, _ = utils.load_checkpoint(checkpoint_path, AE_model)

  generator = models.FlowGenerator(
      n_vocab=len(symbols) + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels,n_speakers=hps.model.n_speaker,gin_channels=256,**hps.model).cuda(n_gpus)
  
  generator.prevec = AE_model.Encoder

  
  for p in generator.prevec.parameters():
    p.requires_grad = False
  

  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, 
    warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

  epoch_str = 1
  global_step = 0
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
    epoch_str += 1
    optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    optimizer_g._update_learning_rate()
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
  
  
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval)
      utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
  global global_step

  generator.train()
  for batch_idx, (x, x_lengths, y, y_lengths, sid) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    sid=sid.cuda(rank,non_blocking=True)
    # Train Generator
    optimizer_g.zero_grad()
    
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths,g=sid, gen=False)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_length = commons.duration_loss(logw, logw_, x_lengths)

    loss_gs = [l_mle, l_length]
    loss_g = sum(loss_gs)
    loss_g.backward()
    grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    optimizer_g.step()
    
    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        (y_gen, *_), *_ = generator(x[:1], x_lengths[:1], g=sid[:1], gen=True)
        audio_logging(y[:1],sid[:1],global_step,hps,writer,batch_idx,'train_org')
        audio_logging(y_gen,sid[:1],global_step,hps,writer,batch_idx,'train')
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_g.item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, (x, x_lengths, y, y_lengths,sid) in enumerate(val_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        sid=sid.cuda(rank,non_blocking=True)
        
        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths,g=sid, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        if batch_idx % hps.train.log_interval == 0:
          (y_gen, *_), *_ = generator(x[:1], x_lengths[:1], g=sid[:1], gen=True)
          audio_logging(y[:1],sid[:1],global_step,hps,writer_eval,batch_idx,'eval_org')
          audio_logging(y_gen,sid[:1],global_step,hps,writer_eval,batch_idx,'eval')
          logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(val_loader.dataset),
            100. * batch_idx / len(val_loader),
            loss_g.item()))
          logger.info([x.item() for x in loss_gs])
           
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    loss_tot = sum(losses_tot)
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))


def audio_logging(audio,sid, epoch, hps, writer,number,type_):
  audio=ap.dynamic_range_decompression(audio)
  mel=audio.detach().cpu()
  mel=mel.numpy()
  mel_basis=librosa.filters.mel(sr=hps.data.sampling_rate, n_fft=hps.data.filter_length, n_mels=hps.data.n_mel_channels)
  covered_mel=librosa.util.nnls(mel_basis,mel)
  cover_audio=librosa.griffinlim(covered_mel,n_iter=60)
  cover_audio=torch.tensor(cover_audio)
  id=sid.detach().cpu()
  id=id.numpy()
  writer.add_audio(type_+"_audio/speakerID_"+str(id)+str(number),cover_audio,epoch,hps.data.sampling_rate)

                           
if __name__ == "__main__":
  main()
