from abc import ABC

from .NGCFLayer import NGCFLayer
import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import SparseTensor
from torch_sparse import mul, fill_diag, sum


def apply_norm(edge_index, add_self_loops=True):
    adj_t = edge_index
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sum(adj_t, dim=1)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    norm_adj_t = mul(adj_t, deg_inv.view(-1, 1))
    return norm_adj_t


class NGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 args,
                 sparse_train,
                 name="LightGCN",
                 **kwargs
                 ):
        super().__init__()
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = args.dim
        self.n_layers = args.layers
        self.l_w = args.weight_decay
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.message_dropout = args.message_dropout
        self.node_dropout = args.node_dropout
        self.normalize = args.normalize
        # self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        row, col = sparse_train.nonzero()
        col = [c + self.num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self.num_users + self.num_items,
                                              self.num_users + self.num_items))
        if self.normalize:
            self.adj = apply_norm(self.adj, add_self_loops=True)
        # self.normalize = args.normalize

        self.Gu = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.num_users, self.embed_k)))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.num_items, self.embed_k)))
        self.Gi.to(self.device)

        propagation_network_list = []
        self.dropout_layers = []

        for layer in range(self.n_layers):
            propagation_network_list.append((NGCFLayer(self.weight_size_list[layer],
                                                       self.weight_size_list[layer + 1]), 'x, edge_index -> x'))
            self.dropout_layers.append(torch.nn.Dropout(p=self.message_dropout))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        # self.softplus = torch.nn.Softplus()
        # self.myparameters = [self.Gu.weight, self.Gi.weight]
        # self.myparameters = [self.Gu, self.Gi]

    def propagate_embeddings(self, adj):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]
        embedding_idx = 0

        for layer in range(self.n_layers):
            all_embeddings += [torch.nn.functional.normalize(self.dropout_layers[embedding_idx](list(
                self.propagation_network.children()
            )[layer](all_embeddings[embedding_idx].to(self.device), adj.to(self.device))), p=2, dim=1)]
            embedding_idx += 1

        all_embeddings = torch.cat(all_embeddings, 1)
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi

    def compute_xui(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, user_id):
        gu, gi = self.propagate_embeddings(self.adj)
        # user_id = Variable(torch.from_numpy(user_id).long(), requires_grad=False).to(self.device)
        # user_emb = self.Gu(user_id)
        # pred = user_emb.mm(self.Gi.weight.t())
        pred = torch.matmul(gu[user_id].to(self.device), torch.transpose(gi.to(self.device), 0, 1))

        return pred

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device())
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = self.adj.to_torch_sparse_coo_tensor().coalesce().indices()
        v = self.adj.to_torch_sparse_coo_tensor().coalesce().values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = SparseTensor(row=i[0],
                           col=i[1],
                           value=v * (1. / (1 - rate)),
                           sparse_sizes=(self.num_users + self.num_items,
                                         self.num_users + self.num_items))

        return out

    def forward(self, user, pos, neg):
        if self.node_dropout > 0:
            sampled_adj = self.sparse_dropout(self.adj,
                                              self.node_dropout,
                                              self.adj.nnz())

        if self.node_dropout > 0:
            adj = sampled_adj
        else:
            adj = self.adj

        gu, gi = self.propagate_embeddings(adj)
        # user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.compute_xui(inputs=(gu[user], gi[pos]))
        xu_neg, _, gamma_i_neg = self.compute_xui(inputs=(gu[user], gi[neg]))
        maxi = torch.nn.LogSigmoid()(xu_pos - xu_neg)
        mf_loss = -1 * torch.mean(maxi)
        # mf_loss = -1 * torch.sum(maxi)
        reg_loss = self.l_w * (1 / 2) * (torch.norm(gu[user]) ** 2
                                         + torch.norm(gi[pos]) ** 2
                                         + torch.norm(gi[neg]) ** 2) / len(user)
        # reg_loss = self.l_w * (1 / 2) * (torch.norm(gu[user]) ** 2
        #                                  + torch.norm(gi[pos]) ** 2
        #                                  + torch.norm(gi[neg]) ** 2)  # / len(user)
        mf_loss += reg_loss

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # return loss.detach().cpu().numpy()

        return mf_loss



