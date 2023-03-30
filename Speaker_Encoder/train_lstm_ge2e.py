import json
import os
import argparse
import math
import sys
import random

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mel_data import Mel_GE2E
from speaker_encoder import LSTM as model
from GE2E import GE2ELoss


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import commons
import utils
import audio_processing as ap

import torch.multiprocessing as mp

global_step=1
config_path='/home/caijb/Desktop/zero_shot_glowtts/Speaker_Encoder/configs/GE2E_base.json'
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
        os.environ["MASTER_PORT"]="12355"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpu,rank=rank)

    if rank == 0:
        logger = utils.get_logger(hps.train.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.train.model_dir)
        writer = SummaryWriter(log_dir=hps.train.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.model_dir, "eval"))

    device = torch.device("cuda:{:d}".format(rank))

    #load_train_dataset
    train_dataset = Mel_GE2E(hps.data.training_files,hps)
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=True,
      batch_size=hps.batch_size, pin_memory=True,
      drop_last=True)
    if rank == 0 :
        eval_dataset =Mel_GE2E(hps.data.validation_files,hps)
        eval_loader = DataLoader(eval_dataset, num_workers=2, shuffle=True,
      batch_size=hps.batch_size, pin_memory=True,
      drop_last=True)

    Conv_Lstm_model = model(input_size= hps.data.n_mel_channels, hidden_size=hps.model.l_hidden, embedding_size=hps.model.embedding_size, num_layers=hps.model.num_layers).to(device)
    ge2e_loss = GE2ELoss(device)


    #mutli_gpu_Set
    if hps.n_gpus>1:
        print("Multi GPU Setting Start")
        Conv_Lstm_model=DistributedDataParallel(Conv_Lstm_model,device_ids=[rank]).to(device)
        print("Multi GPU Setting Finish")

    optimizer = torch.optim.Adam([{'params' : Conv_Lstm_model.parameters()},{ 'params' :ge2e_loss.parameters()}], lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    epoch_str = 1
    global_step = 0

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, device, epoch, hps, Conv_Lstm_model, ge2e_loss, optimizer, train_loader,logger, writer)
            eval(rank, device, epoch, hps, Conv_Lstm_model, ge2e_loss, optimizer, eval_loader, logger, writer_eval)
            if epoch %10 ==0:
                utils.save_checkpoint(Conv_Lstm_model, optimizer, hps.train.learning_rate, epoch, os.path.join(hps.train.model_dir, "EMB_{}.pth".format(epoch)))
                utils.save_checkpoint(ge2e_loss, optimizer, hps.train.learning_rate, epoch, os.path.join(hps.train.model_dir, "GE2E_{}.pth".format(epoch)))
        else : 
            train(rank, device, epoch,  hps, Conv_Lstm_model,ge2e_loss, optimizer, train_loader, None, None)

def train(rank, device, epoch, hps, model,loss_func, optimizer, train_loader, logger, writer):
    global global_step

    model.train()

    for batch_id, mel in enumerate(train_loader):
        tot = 0
        mel = mel.to(device)
        mel = torch.transpose(mel, 3,2)
        optimizer.zero_grad()
        mel = torch.reshape(mel, (hps.batch_size*hps.train.utterance, mel.size(2),mel.size(3)))
        perm = random.sample(range(0, hps.batch_size*hps.train.utterance), hps.batch_size*hps.train.utterance)
        unperm = list(perm)
        for i,j in enumerate(perm):
            unperm[j] = i
        mel = mel[perm]

        emb_vec = model(mel)
        emb_vec = emb_vec[unperm]
        #print(emb_vec.size())
        emb_vec = torch.reshape(emb_vec,(hps.batch_size, hps.train.utterance, emb_vec.size(1)))
        loss= loss_func(emb_vec)

        loss.backward()
        optimizer.step()
        tot = tot + loss
        if rank == 0 :
            if batch_id % hps.train.log_interval == 0:
                logger.info('Train Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, tot/batch_id))

                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                     scalars = {"/Loss" : tot/batch_id}
                )
        global_step += 1

def eval(rank, device, epoch, hps, model,loss_func, optimizer, eval_loader, logger,  writer):
    global global_step

    model.eval()
    with torch.no_grad():
        for batch_id, mel in enumerate(eval_loader):
            tot = 0
            mel = mel.to(device)
            mel = torch.transpose(mel, 3,2)
            optimizer.zero_grad()
            mel = torch.reshape(mel, (hps.batch_size*hps.train.utterance, mel.size(2),mel.size(3)))
            perm = random.sample(range(0, hps.batch_size*hps.train.utterance), hps.batch_size*hps.train.utterance)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel = mel[perm]

            emb_vec = model(mel)
            emb_vec = emb_vec[unperm]
            emb_vec = torch.reshape(emb_vec,(hps.batch_size, hps.train.utterance, emb_vec.size(1)))
            loss= loss_func(emb_vec)
            optimizer.step()
            tot = tot + loss
            if rank == 0 :
                if batch_id % hps.train.log_interval == 0:
                    logger.info('Eval Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, tot/batch_id))

                    utils.summarize(
                        writer = writer,
                        global_step = global_step,
                            scalars = {"/Loss" : tot/batch_id}
                    )
            global_step += 1


if __name__ == "__main__":
    main()