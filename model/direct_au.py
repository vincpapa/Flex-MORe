import torch
import torch.nn as nn
import random
import numpy as np
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

# if torch.cuda.is_available():
#     import torch.cuda as T
# else:
#     import torch as T


class DirectAUModel(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(DirectAUModel, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.l_w = args.weight_decay
        self.gamma = args.gamma
        random_seed = args.seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.device = args.device

        self.user_embeddings = nn.Embedding(num_users, args.dim).to(self.device)
        self.item_embeddings = nn.Embedding(num_items, args.dim).to(self.device)

        self.user_embeddings.weight.data = torch.nn.init.normal_(self.user_embeddings.weight.data, 0, 0.01)
        self.item_embeddings.weight.data = torch.nn.init.normal_(self.item_embeddings.weight.data, 0, 0.01)

        # self.user_embeddings.weight.data = torch.nn.init.xavier_normal_(self.user_embeddings.weight.data)
        # self.item_embeddings.weight.data = torch.nn.init.xavier_normal_(self.item_embeddings.weight.data)

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]


    def forward(self, user_id, pos_id, neg_id):
        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        # neg_emb = self.item_embeddings(neg_id)

        direct_au_loss = self.calculate_loss(user_emb, pos_emb)

        # pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        # neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)

        # tmp = pos_scores - neg_scores

        # maxi = nn.LogSigmoid()(tmp)
        # bpr_loss = -torch.sum(maxi)
        # bpr_loss = -torch.mean(maxi)
        # reg_loss = self.l_w * (torch.norm(user_emb) ** 2
        #                        + torch.norm(pos_emb) ** 2
        #                        + torch.norm(neg_emb) ** 2)  # / len(user_emb)
        # reg_loss = self.l_w * (torch.norm(user_emb) ** 2
        #                        + torch.norm(pos_emb) ** 2) / len(user_emb)

        return direct_au_loss # + reg_loss
        # return bpr_loss

    def custom_forward(self, user_id, pos_id, neg_id):
        user_emb = self.user_embeddings(user_id)
        pos_emb = self.item_embeddings(pos_id)
        neg_emb = self.item_embeddings(neg_id)

        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)

        tmp = pos_scores - neg_scores

        maxi = -1 * nn.LogSigmoid()(tmp)

        b = self.l_w * (1 / 2) * (torch.norm(user_emb, dim=1) ** 2)
        c = self.l_w * (1 / 2) * (torch.norm(pos_emb, dim=1) ** 2)
        d = self.l_w * (1 / 2) * (torch.norm(neg_emb, dim=1) ** 2)

        return torch.mean(maxi + b + c + d), maxi + b + c + d

    def predict(self, user_id):
        # user_id = Variable(torch.from_numpy(user_id).long(), requires_grad=False).to(self.device)
        user_emb = self.user_embeddings(user_id)
        pred = user_emb.mm(self.item_embeddings.weight.t())

        return pred

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, user_e, item_e):
        align = self.alignment(user_e, item_e)
        uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        loss = align + self.gamma * uniform
        return loss


