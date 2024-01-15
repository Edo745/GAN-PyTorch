import torch
import torch.nn as nn

class Discriminator(nn.Module):
    # nc: number of channels of the input image, ndf: number of discriminator filters
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        
        # Set bias to False for convolution layers
        bias = False
        
        # Define convolution layers
        # Parameters: in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=bias)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=bias)
        self.conv4 = nn.Conv2d(ndf*4, 1, 4, 2, 1, bias=bias)

    def forward(self, input):
        # Define activation function (ReLU)
        act = torch.nn.functional.relu
        
        # Apply convolution layers with activation functions
        x = act(self.conv1(input))
        x = act(self.conv2(x))
        x = act(self.conv3(x))
        
        # Apply the final convolution layer without activation function
        x = self.conv4(x)
        
        # Apply sigmoid activation function to normalize output between 0 and 1
        output = torch.sigmoid(x)
        
        # Reshape output
        return output.view(-1, 1).squeeze(1)

