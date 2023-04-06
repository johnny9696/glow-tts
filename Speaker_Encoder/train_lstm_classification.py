import json
import os
import argparse
import math
import sys

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mel_data import MelSID_loader, MelSIDCollate
from speaker_encoder import LSTM_Classification as model


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import commons
import utils
import audio_processing as ap

import torch.multiprocessing as mp

global_step=1
config_path='/home/caijb/Desktop/zero_shot_glowtts/Speaker_Encoder/configs/base_LSTM.json'
save_path='/config.json'

def main():
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.current_device()
    
    #writing config data on the model dir
    with open(config_path,"r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)
    #if no dir 
    if not os.path.exists(hps.train.model_dir):
        os.makedirs(hps.train.model_dir)
    with open(hps.train.model_dir+save_path,'w') as f:
        f.write(data)
    
    torch.manual_seed(hps.train.seed)
    hps.n_gpus = torch.cuda.device_count()
  
    hps.batch_size=int(hps.train.batch_size/hps.n_gpus)
    if hps.n_gpus>1:
        mp.spawn(train_and_eval,nprocs=hps.n_gpus,args=(hps.n_gpus,hps,))
    else:   
        train_and_eval(0,hps.n_gpus,hps)

def train_and_eval(rank,n_gpu, hps):
    global global_step

    if hps.n_gpus>1:
        os.environ["MASTER_ADDR"]="localhost"
        os.environ["MASTER_PORT"]="12359"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpu,rank=rank)

    if rank == 0:
        logger = utils.get_logger(hps.train.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.train.model_dir)
        writer = SummaryWriter(log_dir=hps.train.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.model_dir, "eval"))

    device = torch.device("cuda:{:d}".format(rank))

    #load_train_dataset
    train_dataset = MelSID_loader(hps.data.training_files,hps.data)
    collate_fn = MelSIDCollate(hps.data.slice_length, hps)
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=True,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)
    if rank == 0 :
        eval_dataset = MelSID_loader(hps.data.validation_files,hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=4, shuffle=True,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)

    LSTM_model = model(input_size = hps.data.n_mel_channels , hidden_size = hps.model.hidden_size
     , num_layers=hps.model.num_layers, embedding_size=hps.model.embedding_size, n_speaker=108).to(device)

    #mutli_gpu_Set
    if hps.n_gpus>1:
        print("Multi GPU Setting Start")
        LSTM_model=DistributedDataParallel(LSTM_model,device_ids=[rank]).to(device)
        print("Multi GPU Setting Finish")

    optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    epoch_str = 1
    global_step = 0

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, device, epoch, hps, LSTM_model, optimizer, train_loader,logger, writer)
            eval(rank, device, epoch, hps, LSTM_model, optimizer, eval_loader, logger, writer_eval)

            utils.save_checkpoint(LSTM_model, optimizer, hps.train.learning_rate, epoch, os.path.join(hps.train.model_dir, "G_{}.pth".format(epoch)))
        else : 
            train(rank, device, epoch,  hps, LSTM_model, optimizer, train_loader, None, None)

def train(rank, device, epoch, hps, model, optimizer, train_loader, logger, writer):
    global global_step

    model.train()
    loss = nn.CrossEntropyLoss()

    TP =0
    length =0

    for batch_id,(mel,sid) in enumerate(train_loader):
        mel = mel.to(device)
        sid =sid.to(device)
        

        optimizer.zero_grad()
        mel_hat = model(mel)
        output = loss(mel_hat, sid)
        output.backward()
        optimizer.step()

        if rank == 0 :
            mel_hat_ = torch.argmax(mel_hat, dim = 1)
            TP += torch.sum(mel_hat_ == sid)
            length += len(sid)
            if batch_id % hps.train.log_interval == 0:
                acc = TP/ length
                logger.info('Train Epoch : {}, step : {} , Loss : {}, ACC : {}'.format(epoch, batch_id*epoch, output.item(), acc))

                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                     scalars = {"/Loss" : output.item(),
                     "Acc" : acc}
                )
                TP = 0
                length = 0
        global_step += 1

def eval(rank, device, epoch, hps, model, optimizer, eval_loader, logger,  writer):
    global global_step

    model.eval()
    loss = nn.CrossEntropyLoss()

    TP =0
    length =0

    for batch_id,(mel,sid) in enumerate(eval_loader):
        mel = mel.to(device)
        sid =sid.to(device)
        

        optimizer.zero_grad()
        mel_hat = model(mel)
        output = loss(mel_hat, sid)
        optimizer.step()

        if rank == 0 :
            mel_hat_ = torch.argmax(mel_hat, dim = 1)
            TP += torch.sum(mel_hat_ == sid)
            length += len(sid)
            if batch_id % hps.train.log_interval == 0:
                acc = TP/ length
                logger.info('Eval Epoch : {}, step : {} , Loss : {}, ACC : {}'.format(epoch, batch_id*epoch, output.item(), acc))

                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                     scalars = {"/Loss" : output.item(),
                     "Acc" : acc}
                )
                TP = 0
                length = 0
        global_step += 1

if __name__ == "__main__":
    main()