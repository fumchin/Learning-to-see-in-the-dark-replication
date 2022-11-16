import numpy as np
import torch
from torch import nn
from torchsummary import summary


class Down(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3,stride=1,padding=1):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()   
        )

    def forward(self, x):
        output = self.down(x)
        return output
    
class Bottom(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3,stride=1,padding=1):
        super(Down, self).__init__()

        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.bottom(x)
        return output


class Up(nn.Module):
    def __init__(self, input_channel, up_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels=input_channel, out_channels=up_channel, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=up_channel*2, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x = self.up(x1)
        x2 = x2[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.concatenate([x,x2], dim=1)
        output = self.conv(x)
        return output
        pass

# class Up(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # input shape (6, 6, 9)
#         self.
        

#     def forward(self, x):
#         pass


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(9, 64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, 512)
        self.up2 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.output = nn.Conv2d(64, 27, kernel_size=3, stride=1, padding=1)
        self.pf = nn.PixelShuffle(3)

    def forward(self, x):
        x_1 = self.first_conv(x)
        x_d1 = self.down1(x_1)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_bottom = self.down4(x_d3)
        x_u1 = self.up1(x_bottom, x_d3)
        x_u2 = self.up2(x_u1, x_d2)
        x_u3 = self.up3(x_u2, x_d1)
        x_u4 = self.up4(x_u3, x_1)
        output = self.output(x_u4)
        # output = output.permute(0, 2, 3, 1)
        output = self.pf(output)
        return output

       
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    unet = UNet().cuda()
    summary(unet, (9, 512, 512))