import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1): # nz: input noise dimension, ngf: number of generator filters, nc: number of generated image channels.
        super(Generator, self).__init__()
        # Set bias to False for convolution layers
        bias = False

        # Define convolution transpose layers
        # Parameters: in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0
        self.convt1 = nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=bias)
        self.convt2 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=bias)
        self.convt3 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=bias)
        self.convt4 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=bias)
        self.convt5 = nn.ConvTranspose2d(ngf, nc, 1, 1, 2, bias=bias)

    def forward(self, input):
        # Define activation function (ReLU)
        act = torch.nn.functional.relu
        
        # Apply convolution transpose layers with activation functions
        x = act(self.convt1(input))
        x = act(self.convt2(x))
        x = act(self.convt3(x))
        x = act(self.convt4(x))
        
        # Apply the final convolution layer without activation function
        x = self.convt5(x)
        
        # Apply hyperbolic tangent activation function to normalize output between -1 and 1
        x = torch.tanh(x)
        
        return x
