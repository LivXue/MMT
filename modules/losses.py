import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].float()
        output = -input.gather(2, target.unsqueeze(2)).squeeze(
            2) * mask  # output[i][j] = -input[i][j][target[i][j]] * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
