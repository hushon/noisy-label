import torch
import torch.nn as nn
import torch.nn.functional as F


class overparametrization_loss(nn.Module):
    def __init__(
            self,
            num_examp,
            num_classes=10,
            ratio_consistency = 0,
            ratio_balance = 0
        ):
        super().__init__()
        self.num_classes = num_classes
        # self.config = ConfigParser.get_instance()
        # self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))
        self.init_param()

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        # label = torch.zeros(len(label), self.config['num_classes']).cuda().scatter_(1, label.view(-1,1), 1)
        label = F.one_hot(label, num_classes=self.num_classes).float()

        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)

            ensembled_output = 0.5 * (output + output2).detach()

        else:
            output = outputs

            ensembled_output = output.detach()

        eps = 1e-4

        U_square = torch.clamp(self.u[index]**2 * label , 0, 1)
        V_square = torch.clamp(self.v[index]**2 * (1 - label) , 0, 1)
        E =  U_square - V_square


        self.E = E

        label_one_hot = self.soft_to_hard(output.detach())
        MSE_loss = F.mse_loss(label_one_hot + U_square - V_square, label, reduction='none')

        prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(prediction + U_square - V_square.detach(), min = eps)
        prediction = F.normalize(prediction, p = 1, eps = eps)
        prediction = torch.clamp(prediction, min = eps, max = 1.0)
        ce_loss = -torch.sum((label) * prediction.log(), dim = -1)
        # ce_loss = F.cross_entropy(prediction.log(), label, reduction='none')

        return ce_loss + MSE_loss



    def consistency_loss(self, index, output1, output2):            
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv


    def soft_to_hard(self, x):
        return F.one_hot(x.argmax(dim=1), num_classes=self.num_classes).float()
        # with torch.no_grad():
        #     return (torch.zeros(len(x), self.config['num_classes'])).cuda().scatter_(1, (x.argmax(dim=1)).view(-1,1), 1)