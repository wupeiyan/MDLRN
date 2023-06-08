import torch
import torch.nn as nn
import timm


class CDFI_Branch(nn.Module):
    def __init__(self,channel=3,hidden = 10, dropout=0.2, model_name = 'resnet50', with_att=True):
        super(CDFI_Branch, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  #全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel,hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden,channel),
            nn.Dropout(dropout),
            nn.Sigmoid())

        self.backbone = timm.create_model(model_name, pretrained=False)
        self.backbone.reset_classifier(0, '')
        self.with_att = with_att
        # self.backbone = ResNet50()
    def forward(self, x):
        if self.with_att:
            b, c, _, _ = x.size()# 得到H和W的维度，在这两个维度上进行全局池化
            y = self.gap(x).view(b, c)# Squeeze操作的实现
            y = self.fc(y).view(b, c, 1, 1)# Excitation操作的实现
            x = x * y.expand_as(x)
        out = self.backbone(x)
        # 将y扩展到x相同大小的维度后进行赋权
        return out

class MFA(nn.Module):
    def __init__(self,channel,reduction = 42):
        super(MFA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,  channel// reduction),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid())

    def forward(self, x, a):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y.expand_as(x)
        return y*a

class D_BUS_Net(nn.Module):
    def __init__(self, model_name='resnet50', fuse_att=True, cdfi_att=True, dropout=0.2):
        super(D_BUS_Net, self).__init__()
        #定义模块
        
        self.bus_branch = timm.create_model(model_name, pretrained=False)
        self.bus_branch.reset_classifier(0, '')

        # self.bus_branch = ResNet50()
        last_channel = {
            'resnet50': 2048,
            'resnet34' : 512, 
            'resnet18' : 512, 
            'efficientnet_b0' : 1280, 
            'convnext_small': 768
        }
        self.cdfi_branch = CDFI_Branch(model_name=model_name, with_att=cdfi_att, dropout=dropout)
        last_conv_channel = last_channel[model_name]
        # last_conv_channel = 1000

        self.MFA = MFA(last_conv_channel)  #resnet50 224决定了这里2048

        self.flatten = nn.Flatten()

        
        self.bus_cls = nn.Sequential(
            nn.Linear(last_conv_channel*7*7, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.cdfi_cls = nn.Sequential(
            nn.Linear(last_conv_channel*7*7, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Linear(last_conv_channel*2*7*7, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        self.fuse_att = fuse_att


    def forward(self,bus_input,cdfi_input):
        bus_output = self.bus_branch(bus_input)
        cdfi = self.cdfi_branch(cdfi_input)
        if self.fuse_att:
            cdfi_output = self.MFA(bus_output,cdfi)
        else:
            cdfi_output = cdfi
        bus_output = self.flatten(bus_output)
        cdfi_output = self.flatten(cdfi_output)
        out = torch.cat([bus_output,cdfi_output],dim = 1)
        bus_aux_out = self.bus_cls(bus_output)
        cdfi_aux_out = self.cdfi_cls(cdfi_output)
        out = self.cls(out)
        return out ,bus_aux_out,cdfi_aux_out



if __name__ == '__main__':
    input1 = torch.randn((1, 3, 224, 224))
    input2 = torch.randn((1, 3, 224, 224))

    out1 ,out2 , out3 = D_BUS_Net('convnext_small')(input1,input2)
    print(out1,out2,out3)
