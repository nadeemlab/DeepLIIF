# adapted from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,innermost=False,outermost=False):
        super(conv_block,self).__init__()
        if outermost:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=True),
                nn.LeakyReLU(0.2, True),
            )
        elif innermost:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4,stride=2,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, True),
            )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,innermost=False,outermost=False):
        super(up_conv,self).__init__()
        use_bias=False
        if outermost:
            self.up = nn.Sequential(
                          nn.ConvTranspose2d(ch_in * 2, ch_out,
                                        kernel_size=4, stride=2,
                                        padding=1),
                          nn.Tanh())
        elif innermost:
            self.up = nn.Sequential(
                          nn.ConvTranspose2d(ch_in, ch_out,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                          nn.BatchNorm2d(ch_out),
                          nn.ReLU(True))
        else:
            self.up = nn.Sequential(
                          nn.ConvTranspose2d(ch_in * 2, ch_out,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                          nn.BatchNorm2d(ch_out),
                          nn.ReLU(True))
                                        

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,outermost=True)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=512)
        self.Conv6 = conv_block(ch_in=512,ch_out=512)
        self.Conv7 = conv_block(ch_in=512,ch_out=512)
        self.Conv8 = conv_block(ch_in=512,ch_out=512,innermost=True)
        #self.Conv9 = conv_block(ch_in=512,ch_out=512,innermost=True)
        
        self.Up8 = up_conv(ch_in=512,ch_out=512,innermost=True)
        self.Att8 = Attention_block(F_g=512,F_l=512,F_int=512)
        
        self.Up7 = up_conv(ch_in=512,ch_out=512)
        self.Att7 = Attention_block(F_g=512,F_l=512,F_int=512)
        
        self.Up6 = up_conv(ch_in=512,ch_out=512)
        self.Att6 = Attention_block(F_g=512,F_l=512,F_int=512)
        
        self.Up5 = up_conv(ch_in=512,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=512)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        
        self.Up1 = up_conv(ch_in=64,ch_out=output_ch,outermost=True)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        x6 = self.Conv6(x5)
        x7 = self.Conv7(x6)
        x8 = self.Conv8(x7)
        #x9 = self.Conv9(x8)
        
        #d9 = self.Up
        d8 = self.Up8(x8)
        x7 = self.Att8(g=d8,x=x7)
        d8 = torch.cat((x7,d8),dim=1)

        d7 = self.Up7(d8)
        x6 = self.Att7(g=d7,x=x6)
        d7 = torch.cat((x6,d7),dim=1)
        
        d6 = self.Up6(d7)
        x5 = self.Att6(g=d6,x=x5)
        d6 = torch.cat((x5,d6),dim=1)
        
        d5 = self.Up5(d6)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        
        
        d4 = self.Up4(d5) # x4: [2, 512, 4, 4], d4: [2, 256, 4, 4]
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)

        d1 = self.Up1(d2)

        return d1

