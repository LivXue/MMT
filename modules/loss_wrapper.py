import torch

import modules.losses as losses


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = losses.LanguageModelCriterion()

    def forward(self, conv_feats, conv_masks, labels, masks, src1, src2, src3, src4):
        outputs = self.model(conv_feats, conv_masks, labels[..., :-1], src1, src2, src3, src4)
        loss = self.crit(outputs, labels[..., 1:], masks[..., 1:])
        return loss
