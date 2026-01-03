import numpy as np
import torch
import pickle
import random
import os
import json
import scipy.sparse as sp
import csv

def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance", is_text=True) -> np.array:

    A = np.zeros([int(num_nodes), int(num_nodes)])
    if is_text:
        with open(distance_file, "r") as f_d:
            f_d.readline()
            reader = csv.reader(f_d)
            for item in reader:
                if len(item) != 3:
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if graph_type == "connect":
                    A[i, j], A[j, i] = 1., 1.
                    # A[i, j] = 1.
                elif graph_type == "distance":
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")
    else:
        A = np.load(distance_file)

    return A


def transform_matrix(matrix, A):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != 0:
                A[i][j] = 1
                A[rows - 1 - i][cols - 1 - j] = 1

    return A


def process_graph(graph_data):
    N = graph_data.size(0)
    matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)
    graph_data += matrix_i

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)
    degree_matrix = degree_matrix.pow(-1)
    degree_matrix[degree_matrix == float("inf")] = 0.

    degree_matrix = torch.diag(degree_matrix)

    return torch.mm(degree_matrix, graph_data)



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    # if any(field in pkl_filename for field in ["PEMSBAY", "METRLA"]):
    _, _, adj_mx = load_pickle(pkl_filename)
    # if any(field in pkl_filename for field in ["03", "04", "08", "07"]):
    #     adj_mx = load_pickle(pkl_filename)
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    return adj_mx


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """adj-->[N,N]
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)
    Args:
        adj (np.ndarray): Adjacent matrix A
    Returns:
        np.matrix: Transition matrix P
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()
    return prob_matrix


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels, null_val):
    if not isinstance(preds, torch.Tensor):
        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)
        



def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


if __name__=='__main__':
    path = '../data/PEMSBAY/adj.pkl'
    out = np.load('../data/PEMSBAY/adj.npy')