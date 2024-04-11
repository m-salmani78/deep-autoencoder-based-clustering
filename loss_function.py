from torch import Tensor
import torch.nn as nn
import torch.utils.data

beta = 0.00001

class CustomAutoencoderLoss(nn.modules.loss._Loss):
    def __init__(self, mse_weight: Tensor):
        super(CustomAutoencoderLoss, self).__init__()
        self.mse_weight = mse_weight

    def forward(self, input: Tensor, target: Tensor, parameters):
        # Wm weighted MSE loss
        L_cmse = torch.sum(self.mse_weight * ((input - target) ** 2)) / len(self.mse_weight)

        # L2 regularization
        l2_regularization = sum(torch.norm(param, p=2) for param in parameters)

        loss = L_cmse + beta * l2_regularization

        return loss
