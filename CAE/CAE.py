import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, encoder_dim=1,
    hidden_1dim=3,
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        self.kernel=kernel

        self.conv2d_layer_1=nn.Conv2d(self.encoder_dim,self.hidden_dim_1,kernel_size=kernel)
        self.conv2d_layer_2=nn.Conv2d(self.hidden_dim_1,self.hidden_dim_1+2,kernel_size=kernel)
        self.conv2d_layer_3=nn.Conv2d(self.hidden_dim_1+2,self.hidden_dim_1+6,kernel_size=kernel)
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
    encoder_dim,
    hidden_1dim,
    kernel=5):
    
        super().__init__()
        self.Tconv2d_layer1=nn.ConvTranspose2d(hidden_1dim+6, hidden_1dim+2, kernel_size=kernel)
        self.Tconv2d_layer2=nn.ConvTranspose2d(hidden_1dim+2, hidden_1dim, kernel_size=kernel)
        self.Tconv2d_layer3=nn.ConvTranspose2d(hidden_1dim, encoder_dim, kernel_size=kernel)
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
    kernel=5):
        super().__init__()
        self.encoder_dim=encoder_dim
        self.hidden_dim_1=hidden_1dim
        

        #convolution autoencoder
        self.encoder=Encoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, kernel=kernel)
        self.decoder=Decoder(encoder_dim=self.encoder_dim, hidden_1dim=self.hidden_dim_1, kernel=kernel)

    def forward(self,mel,classification=False):
        mel= self.encoder(mel)
        if classification == False:
            mel=self.decoder(mel)
        return mel
    
    def get_vector(self,mel):
        return self.encoder(mel)


