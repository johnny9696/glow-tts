import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, encoder_dim=80,
    hidden_1dim=25,
    hidden_2dim = 5,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2 = hidden_2dim
        self.kernel=kernel

        self.conv2d_layer_1=nn.Conv1d(self.encoder_dim,self.hidden_dim_1,kernel_size=kernel)
        self.conv2d_layer_2=nn.Conv1d(self.hidden_dim_1,self.hidden_dim_2,kernel_size=kernel)
        self.conv2d_layer_3=nn.Conv1d(self.hidden_dim_2,1,kernel_size=kernel)
        self.relu=nn.ReLU()

    def forward(self,mel):
        x=self.conv2d_layer_1(mel)
        x=self.relu(x)
        x=self.conv2d_layer_2(x)
        x=self.relu(x)
        x=self.conv2d_layer_3(x)
        x=self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
    encoder_dim=80,
    hidden_1dim=25,
    hidden_2dim =5,
    kernel=5):
    
        super().__init__()
        self.Tconv2d_layer1=nn.ConvTranspose1d(1, hidden_2dim, kernel_size=kernel)
        self.Tconv2d_layer2=nn.ConvTranspose1d(hidden_2dim, hidden_1dim, kernel_size=kernel)
        self.Tconv2d_layer3=nn.ConvTranspose1d(hidden_1dim, encoder_dim, kernel_size=kernel)
        self.relu=nn.ReLU()

    def forward(self,z):
        z=self.Tconv2d_layer1(z)
        z=self.relu(z)
        z=self.Tconv2d_layer2(z)
        z=self.relu(z)
        z=self.Tconv2d_layer3(z)
        z=self.relu(z)
    
        return z


class Convolution_Auto_Encoder(nn.Module):
    def __init__(self,
    encoder_dim,
    hidden_1dim,
    hidden_2dim,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.hidden_dim_2=hidden_2dim
        

        #convolution autoencoder
        self.encoder=Encoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, kernel=kernel)
        self.decoder=Decoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, kernel=kernel)

    def forward(self,mel):
        mel= self.encoder(mel)
        mel=self.decoder(mel)
        return mel
    
    def get_vector(self,mel):
        return self.encoder(mel)



