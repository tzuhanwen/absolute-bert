from collections import defaultdict
import torch


def format_losses(loss, clip_value=None):
    # 判斷是否為多重 loss（dict）
    is_multiloss = isinstance(loss, dict)

    if is_multiloss:
        total_loss = sum(loss.values())
    else:
        total_loss = loss

    final_loss = torch.clip(total_loss, max=clip_value) if clip_value is not None else total_loss

    # 建立輸出 dict
    loss_dict = {
        "loss": total_loss.item(),
        "final_loss": final_loss.item()
    }

    if is_multiloss:
        loss_dict |= {key: value.item() for key, value in loss.items()}

    return final_loss, loss_dict


class MultiLossAverager:
    def __init__(self):
        self.total_loss = defaultdict(float)
        self.total_count = 0

    def update(self, loss_dict, batch_size):
        for key, value in loss_dict.items():
            self.total_loss[key] += value * batch_size
        self.total_count += batch_size

    def compute(self):
        if self.total_count == 0:
            return {key: 0.0 for key in self.total_loss}
        return {key: total / self.total_count for key, total in self.total_loss.items()}

    def reset(self):
        self.total_loss = defaultdict(float)
        self.total_count = 0