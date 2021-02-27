import pretrainedmodels
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch


class Model(nn.Module):
    def __init__(self, pretrained, model_name, teams_dic_len, players_dic_len):
        """ model_name = ['resnet34', 'resnet50', 'mobilenetv2']  """
        super(Model, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__[model_name](pretrained=None)
        # print(self.model)
        self.fc1 = nn.Linear(512, teams_dic_len)  # For Teams class
        self.fc2 = nn.Linear(512, players_dic_len)  # For players class

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label1 = self.fc1(x)
        label2 = self.fc2(x)
        return {"label1": label1, "label2": label2}
