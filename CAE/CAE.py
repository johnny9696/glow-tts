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
        self.pooling = nn.MaxPool2d(kernel_size = kernel,stride=1, return_indices = True)
        self.relu=nn.ReLU()

    def forward(self,mel):
        x=self.conv2d_layer_1(mel)
        x=self.relu(x)
        x, indice1=self.pooling(x)
        x=self.conv2d_layer_2(x)
        x=self.relu(x)
        x, indice2=self.pooling(x)
        x=self.conv2d_layer_3(x)
        x=self.relu(x)
        x, indice3=self.pooling(x)

        return x, [indice1, indice2, indice3]

class Decoder(nn.Module):
    def __init__(self,
    encoder_dim,
    hidden_1dim,
    kernel=5):
    
        super().__init__()
        self.Tconv2d_layer1=nn.ConvTranspose2d(hidden_1dim+6, hidden_1dim+2, kernel_size=kernel)
        self.Tconv2d_layer2=nn.ConvTranspose2d(hidden_1dim+2, hidden_1dim, kernel_size=kernel)
        self.Tconv2d_layer3=nn.ConvTranspose2d(hidden_1dim, encoder_dim, kernel_size=kernel)
        self.unpooling = nn.MaxUnpool2d(kernel_size=kernel,stride=1)
        self.relu=nn.ReLU()

    def forward(self,z, indice):
        z=self.unpooling(z,indice[2])
        z=self.Tconv2d_layer1(z)
        z=self.relu(z)
        z=self.unpooling(z, indice[1])
        z=self.Tconv2d_layer2(z)
        z=self.relu(z)
        z=self.unpooling(z, indice[0])
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
        mel, indice = self.encoder(mel)
        if classification == False:
            mel=self.decoder(mel, indice)
        return mel
    
    def get_vector(self,mel):
        return self.encoder(mel)


