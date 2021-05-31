import torch

class OnlineDistillationLoss(torch.nn.Module):
    def __init__(self):
        super(OnlineDistillationLoss, self).__init__()

    def forward(self, outputs, target):
        cee = torch.nn.CrossEntropyLoss()
        kld = torch.nn.KLDivLoss()

        cnt_outputs = outputs.shape[0]

        if cnt_outputs < 2:
            raise ValueError(f'online distillation loss must have at least 2 outputs')

        #cpy_outputs = torch.stack([output.detach().clone() for output in outputs])
        cpy_outputs = outputs.detach().clone()
        losses = []

        for i in range(cnt_outputs):
            #print(cpy_outputs.shape)
            #other_outputs = torch.cat([cpy_outputs[0:i], cpy_outputs[i + 1:]])
            #print(other_outputs.shape)
            #print(torch.mean(other_outputs, dim=0).shape)
            #print(outputs[i].shape)
            #print(target.shape)
            #loss_cee = cee(outputs[i], target)
            #loss_kd = kld(torch.mean(other_outputs, dim=0), outputs[i])

            other_outputs = torch.cat([cpy_outputs[0:i], cpy_outputs[i+1:]])
            loss = cee(outputs[i], target) + kld(torch.mean(other_outputs, dim=0), outputs[i])
            losses.append(loss)

        loss = losses[0] + losses[1]

        for i in range(2, cnt_outputs):
            loss += losses[i]

        return loss
