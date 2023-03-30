import torch
import torch.nn as nn
from torch import Tensor

class LSTM_Classification(nn.Module) :
    def __init__(self,
    input_size = 80,
    hidden_size= 512,
    num_layers = 3,
    embedding_size= 256,
    n_speaker=0):
        super().__init__()
        self.layer3_lstm = nn.LSTM(input_size = input_size , hidden_size = hidden_size, num_layers = num_layers, batch_first =True)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.linear1 =nn.Linear(embedding_size, n_speaker)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, mel):
        mel = torch.transpose(mel, 1, 2) #[b,frames, n_mels]
        mel, (_,_)=self.layer3_lstm(mel) #[b, frames, hidden]
        #mel = torch.sum(mel, dim =1)/mel.size(1) #[b,hidden]
        mel = self.linear(mel[:,-1])
        mel = self.relu(mel)
        mel = self.linear1(mel)
        mel = self.softmax(mel)
        return mel
        
    def get_vector(self, mel):
        mel = torch.transpose(mel, 1, 2) #[b,frames, n_mels]
        mel, (_,_)=self.layer3_lstm(mel) #[b, frames, hidden]
        #mel = torch.sum(mel, dim =1)/mel.size(1) #[b,hidden]
        mel = self.linear(mel[:,-1])
        mel = self.relu(mel)
        return mel

class LSTM(nn.Module) :
    def __init__(self,
    input_size = 80,
    hidden_size= 512,
    embedding_size =256,
    num_layers = 3):
        super().__init__()
        self.layer3_lstm = nn.LSTM(input_size = input_size , hidden_size = hidden_size, num_layers = num_layers, batch_first =True)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()
    
    def forward(self, mel):
        mel, (_,_)=self.layer3_lstm(mel) #[b, frames, hidden]
        #mel = torch.sum(mel, dim =1)/mel.size(1) #[b,hidden]
        mel = self.linear(mel[:,-1])
        return self.relu(mel)

class Encoder(nn.Module):
    def __init__(self, encoder_dim=430,
    hidden_1dim= 256,
    hidden_2dim =  64,
    hidden_3dim = 16,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2 = hidden_2dim
        self.kernel=kernel

        self.conv2d_layer_1=nn.Conv1d(self.encoder_dim,self.hidden_dim_1,kernel_size=kernel)
        self.conv2d_layer_2=nn.Conv1d(self.hidden_dim_1,self.hidden_dim_2,kernel_size=kernel)
        self.conv2d_layer_3=nn.Conv1d(self.hidden_dim_2,hidden_3dim,kernel_size=kernel)
        self.relu=nn.ReLU()

    def forward(self,mel):
        x=self.conv2d_layer_1(mel)
        x=self.relu(x)
        x=self.conv2d_layer_2(x)
        x=self.relu(x)
        x=self.conv2d_layer_3(x)
        x=self.relu(x)
        return x


class Convolution_LSTM_classification(nn.Module):
    def __init__(self,
    encoder_dim=430,
    hidden_dim1=256,
    hidden_dim2=64,
    hiddem_dim3=16,
    l_hidden = 768,
    num_layers = 3,
    input_size = 80,
    embedding_size =256,
    kernel=5,
    n_speaker = 0):
        super().__init__()
        self.encoder_dim =encoder_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hiddem_dim3
        self.kernel  = kernel

        self.l_hidden = l_hidden
        self.num_layers = num_layers
        self.input_size = input_size-3*(kernel-1)
        
        self.encoder = Encoder(encoder_dim= self.encoder_dim, hidden_1dim=self.hidden_dim1, hidden_2dim= self.hidden_dim2, hidden_3dim=self.hidden_dim3, kernel=self.kernel )
        self.LSTM = LSTM(input_size=self.input_size, hidden_size=self.l_hidden, num_layers=self.num_layers, embedding_size=embedding_size)
        self.FC = nn.Linear(embedding_size, n_speaker)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, mel):
        mel = torch.transpose(mel,2,1)
        mel = self.encoder(mel)
        mel = self.LSTM(mel)
        mel = self.FC(mel)
        mel = self.softmax(mel)
        return mel
    
    def get_vector(self, mel):
        mel = torch.transpose(mel,2,1)
        mel = self.encoder(mel)
        mel = self.LSTM(mel)

        return mel

class Convolution_LSTM_cos(nn.Module):
    def __init__(self,
    encoder_dim=430,
    hidden_dim1=256,
    hidden_dim2=64,
    hiddem_dim3=16,
    l_hidden = 768,
    num_layers = 3,
    input_size = 80,
    embedding_size =256,
    kernel=5):
        super().__init__()
        self.encoder_dim =encoder_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hiddem_dim3
        self.kernel  = kernel

        self.l_hidden = l_hidden
        self.num_layers = num_layers
        self.input_size = input_size-3*(kernel-1)
        
        self.encoder = Encoder(encoder_dim= self.encoder_dim, hidden_1dim=self.hidden_dim1, hidden_2dim= self.hidden_dim2, hidden_3dim=self.hidden_dim3, kernel=self.kernel )
        self.LSTM = LSTM(input_size=self.input_size, hidden_size=self.l_hidden, num_layers=self.num_layers, embedding_size=embedding_size)

    def forward(self, mel):
        mel = torch.transpose(mel,2,1)
        mel = self.encoder(mel)
        mel = self.LSTM(mel)
        return mel
    



