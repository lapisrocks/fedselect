import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class MaskLocalAltSGD(optim.Optimizer):
    def __init__(self, params, mask: OrderedDict = None, lr=0.01):
        """Implements SGD with alternating updates based on a binary mask of parameters."""
        # require params is named parameters
        # assert isinstance(params, list) and len(params) == 1
        self.mask: list[torch.Tensor] = [value.long() for key, value in mask.items()]
        self.names: list[torch.Tensor] = [key for key, value in mask.items()]
        self.named_mask: OrderedDict = mask
        self._toggle = True
        defaults = dict(lr=lr, _toggle=True)

        if mask is None:
            raise ValueError("MaskLocalAltSGD requires a mask")
        super(MaskLocalAltSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskLocalAltSGD, self).__setstate__(state)

    def toggle(self):
        self._toggle = not self._toggle

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        # update parameters
        for group in self.param_groups:
            step = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                # assert that p does not contain nan
                assert torch.isnan(p).sum() == 0, "parameter contains nan"
                # get name of parameter
                mask = self.mask[step]
                # update parameter
                if mask is not None:
                    if self._toggle:
                        p.data.add_(mask * p.grad.data, alpha=-group["lr"])
                    else:
                        p.data.add_((1 - mask) * p.grad.data, alpha=-group["lr"])
                else:
                    p.data.add_(-group["lr"], p.grad.data)
                step += 1
        return loss


def local_alt(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    clip_grad_norm=False,
    max_grad_norm=3.50,
):
    assert isinstance(optimizer, MaskLocalAltSGD), "optimizer must be MaskLocalAltSGD"
    avg_loss_1 = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss_1 += loss.item()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    avg_loss_1 /= len(data_loader)
    optimizer.toggle()

    avg_loss_2 = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss_2 += loss.item()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    avg_loss_2 /= len(data_loader)
    optimizer.toggle()

    train_loss = (avg_loss_1 + avg_loss_2) / 2
    return train_loss


if __name__ == "__main__":
    pass
