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
from StackedAE import Convolution_Auto_Encoder as ACE
from StackedAE import Convolution_AE_Classification as model
from sid_list import sid as sid_list


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import commons
import utils
import audio_processing as ap

import torch.multiprocessing as mp

global_step=1
config_path='/media/caijb/data_drive/autoencoder/log/kernel5/config.json'
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
        os.environ["MASTER_PORT"]="22545"
        dist.init_process_group(backend='nccl',init_method='env://',world_size=n_gpu,rank=rank)

    if rank == 0:
        logger = utils.get_logger(hps.train.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.train.model_dir)
        writer = SummaryWriter(log_dir=hps.train.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.model_dir, "eval"))

    device = torch.device("cuda:{:d}".format(rank))

    #load_train_dataset need to fix
    train_dataset = MelSID_loader(hps.data.training_files,hps.data)
    collate_fn = MelSIDCollate(hps.data.slice_length, hps)
    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)
    if rank == 0 :
        eval_dataset = MelSID_loader(hps.data.validation_files,hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn)

    AE_model = ACE(encoder_dim=hps.model.encoder_dim, hidden_1dim=hps.model.hidden_dim1,
    hidden_2dim=hps.model.hidden_dim2, kernel=hps.model.kernel).to(device)

    #load model dict
    checkpoint_path = "/media/caijb/data_drive/autoencoder/log/kernel5"
    checkpoint_path = utils.latest_checkpoint_path(checkpoint_path)
    AE_model, _, _, _ = utils.load_checkpoint(checkpoint_path, AE_model)

    CAC_model = model (encoder_dim=hps.model.encoder_dim, hidden_1dim=hps.model.hidden_dim1,
    hidden_2dim=hps.model.hidden_dim2, kernel=hps.model.kernel,n_speaker= hps.model.output_channel, hps = hps).to(device)

    CAC_model.encoder = AE_model.encoder
    """
    for p in CAC_model.encoder.parameters():
        p.requires_grad = False
    """
    #mutli_gpu_Set
    if hps.n_gpus>1:
        print("Multi GPU Setting Start")
        CAC_model=DistributedDataParallel(CAC_model,device_ids=[rank]).to(device)
        print("Multi GPU Setting Finish")
    
    optimizer = torch.optim.Adam(CAC_model.parameters(), lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    epoch_str = 1
    global_step = 0

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, device, epoch, hps, CAC_model, optimizer, train_loader,logger, writer)
            eval(rank, device, epoch, hps, CAC_model, optimizer, eval_loader, logger, writer_eval)

            utils.save_checkpoint(CAC_model, optimizer, hps.train.learning_rate, epoch, os.path.join(hps.train.model_dir, "G_{}.pth".format(epoch)))
        else : 
            train(rank, device, epoch,  hps, CAC_model, optimizer, train_loader, None, None)

def metric(table, n_class):
    metric_count=[[0 for i in  range(0,4)] for j in range(n_class)]
    acc_table=[]
    precision_table = []
    recall_table = []
    f1_table = []
    ramda = 1e-10

    for i in range(0, len(table)):
        for j in range(0, len(table)):
            if i == j :
                metric_count[i][0] += table[i][j]
            else :
                metric_count[i][1] += table[i][j]
                metric_count[j][2] += table[i][j]
    
    for i in metric_count:
        try:
            acc_table.append(float(i[0])/(float(i[0])+float(i[1])+float(i[2])+ramda))
            recall_table.append(float(i[0])/(float(i[0])+float(i[2])+ramda))
            precision_table.append(float(i[0])/(float(i[0])+float(i[1])+ramda))
            f1_table.append(recall_table[-1]*precision_table[-1]/(precision_table[-1]+recall_table[-1]+ramda))
        except :
            raise Exception(i)

    return acc_table, precision_table, recall_table, f1_table

def train(rank, device, epoch, hps, model, optimizer, train_loader, logger, writer):
    global global_step

    model.train()
    loss = nn.CrossEntropyLoss().to(device)
   
    metric_table = [[0 for x in range(0,hps.model.output_channel)] for i in range(0, hps.model.output_channel)]
    for batch_id,(mel_padded, sid) in enumerate(train_loader):
        mel_padded = mel_padded.to(device)
        sid = sid.to(device)
        
        optimizer.zero_grad()
        #label_hat shape [batch, output class]
        label_hat = model(mel_padded)
        output = loss(label_hat, sid)
        output.backward()
        optimizer.step()

        if rank == 0 :
            label_hat_ = torch.argmax(label_hat, dim = 1)
            #print(sid , label_hat_)
            for i in range(0, len(sid)):
                metric_table[sid[i]][label_hat_[i]] += 1
            
            if batch_id % hps.train.log_interval == 0:
                acc,pre,recall,f1 = metric(metric_table, hps.model.output_channel)
                logger.info('Train Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, output.item()))
                logger.info("Acc    Pre    Recall    F1-Score")
                logger.info("=================================================")
                
                m_acc = sum(acc)/len(acc)
                #for i in range(len(acc)):
                #    logger.info("{}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(i,acc[i],pre[i], recall[i], f1[i]))
                m_acc = sum(acc)/len(acc)
                m_pre = sum(pre)/len(pre)
                m_recall = sum(recall)/len(recall)
                m_f1 = sum(f1)/len(f1)
                logger.info("{:.4f}  {:.4f}  {:.4f}  {:.4f}".format(m_acc, m_pre, m_recall, m_f1))
                utils.summarize(
                    writer = writer,
                    global_step = global_step,
                     scalars = {"/Loss" : output.item(),
                     "/Metric/Macro_ACC" : m_acc,
                     "/Metric/Macro_Precision" : m_pre,
                     "/Metric/Macro_Recall" : m_recall,
                     "/Metric/Macro_F1" : m_f1}
                )
                metric_table = [[0 for x in range(0,hps.model.output_channel)] for i in range(0, hps.model.output_channel)]
        global_step += 1


def eval(rank, device, epoch, hps, model, optimizer, eval_loader, logger,  writer):
    global global_step

    model.eval()
    loss = nn.CrossEntropyLoss().to(device)
    
    metric_table = [[0 for x in range(0, hps.model.output_channel)] for i in range(0, hps.model.output_channel)]
    with torch.no_grad():
        for batch_id,(mel_padded, sid) in enumerate(eval_loader):
            mel_padded = mel_padded.to(device)
            sid = sid.to(device)
        
            optimizer.zero_grad()
            #label_hat shape [batch, output class]
            label_hat = model(mel_padded)
            output = loss(label_hat, sid)
            optimizer.step()

            if rank == 0 :
                label_hat_ = torch.argmax(label_hat, dim = 1)
                for i in range(0, len(sid)):
                    metric_table[sid[i]][label_hat_[i]] += 1
            
                if batch_id % hps.train.log_interval == 0:
                    acc,pre,recall,f1 = metric(metric_table, hps.model.output_channel)
                    logger.info('Eval Epoch : {}, step : {} , Loss : {}'.format(epoch, batch_id*epoch, output.item()))
                    logger.info("Acc    Pre    Recall    F1-Score")
                    logger.info("=================================================")
                    
                    #for i in range(len(acc)):
                    #    logger.info("{}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(i,acc[i],pre[i], recall[i], f1[i]))
                    m_acc = sum(acc)/len(acc)
                    m_pre = sum(pre)/len(pre)
                    m_recall = sum(recall)/len(recall)
                    m_f1 = sum(f1)/len(f1)
                    logger.info("{:.4f}  {:.4f}  {:.4f}  {:.4f}".format(m_acc, m_pre, m_recall, m_f1))
                    utils.summarize(
                        writer = writer,
                        global_step = global_step,
                         scalars = {"/Loss" : output.item(),
                         "/Metric/Macro_ACC" : m_acc,
                         "/Metric/Macro_Precision" : m_pre,
                         "/Metric/Macro_Recall" : m_recall,
                         "/Metric/Macro_F1" : m_f1}
                    )
                    metric_table = [[0 for x in range(0, hps.model.output_channel)] for i in range(0, hps.model.output_channel)]
            global_step += 1

if __name__ == "__main__":
    main()