import math
import numpy as np
import torch


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def idcg_k(k):
    res = np.sum(1.0 / np.log2(np.arange(2, k + 2)))
    return res if res > 0 else 1.0


def ndcg_k_torch(actual, predicted, topk, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_users = len(actual)
    max_act_len = max(len(a) for a in actual)

    actual_mat = torch.full((num_users, max_act_len), -1, dtype=torch.long, device=device)
    pred_mat = torch.full((num_users, topk), -1, dtype=torch.long, device=device)

    for i, (a, p) in enumerate(zip(actual, predicted)):
        actual_mat[i, :len(a)] = torch.tensor(a, dtype=torch.long, device=device)
        pred_mat[i, :min(len(p), topk)] = torch.tensor(p[:topk], dtype=torch.long, device=device)

    relevance = (pred_mat.unsqueeze(-1) == actual_mat.unsqueeze(1)).any(-1).float()

    log_pos = 1.0 / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
    dcg = (relevance * log_pos).sum(dim=1)

    ideal_len = torch.tensor([min(len(a), topk) for a in actual], device=device)
    cum_log_pos = torch.cumsum(log_pos, dim=0)  # shape (topk,)
    idx = torch.clamp(ideal_len, min=1) - 1
    idcg = torch.where(
        ideal_len > 0,
        cum_log_pos[idx],
        torch.ones_like(ideal_len, dtype=torch.float)  # evitiamo /0 per chi non ha item rilevanti
    )

    ndcg = (dcg / idcg).mean().item()
    return ndcg


def ndcg_k(actual, predicted, topk):
    num_users = len(actual)

    # Step 1: Padding e costruzione array 2D
    pred_mat = np.full((num_users, topk), -1, dtype=int)  # -1 = item fittizio
    rel_mat = np.zeros((num_users, topk), dtype=np.float32)

    for i, (a, p) in enumerate(zip(actual, predicted)):
        k = min(topk, len(p))
        pred_mat[i, :k] = p[:k]
        actual_set = set(a)
        rel_mat[i, :k] = [1.0 if item in actual_set else 0.0 for item in p[:k]]

    log_pos = 1.0 / np.log2(np.arange(2, topk + 2))
    dcg = np.sum(rel_mat * log_pos, axis=1)

    ideal_len = np.array([min(len(a), topk) for a in actual])
    idcg = np.array([np.sum(log_pos[:l]) if l > 0 else 1.0 for l in ideal_len])

    ndcg = dcg / idcg
    return np.mean(ndcg)

def ndcg_k_mid(actual, predicted, topk):
    total_ndcg = 0.0
    for user_actual, user_pred in zip(actual, predicted):
        actual_set = set(user_actual)
        pred_topk = user_pred[:topk]

        relevance = np.array([1 if item in actual_set else 0 for item in pred_topk])

        denom = np.log2(np.arange(2, len(relevance) + 2))
        dcg = np.sum(relevance / denom)

        ideal_k = min(topk, len(actual_set))
        idcg = idcg_k(ideal_k)

        total_ndcg += dcg / idcg

    return total_ndcg / len(actual)

'''
def ndcg_k_old(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k_old(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k_old(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


# build ndcg_list for each user
def ndcg_list(actual, predicted, topk):
    res = []
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k_old(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res.append(dcg_k / idcg)
        # res.append(dcg_k)
    return res
'''


if __name__ == '__main__':
    actual = [[1, 2], [3, 4, 5]]
    predicted = [[10, 20, 1, 30, 40], [10, 3, 20, 4, 5]]
    print(ndcg_k(actual, predicted, 5))