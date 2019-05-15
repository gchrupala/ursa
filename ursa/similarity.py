import torch
import torch.nn as nn
import torch.nn.functional as F
import Levenshtein as L

class LearnedSimilarity(nn.Module):

    def __init__(self, size_in):
        super(LearnedSimilarity, self).__init__()
        self.size_in = size_in
        self.W = nn.Linear(self.size_in, self.size_in)
        

    def forward(self, u, v):
        score = u @ self.W(v).t()
        alpha = torch.softmax(score.view(-1), dim=0).view(score.size())
        sim = cosine_matrix(u, v)
        return (alpha * sim).sum()

class WeightedAverage(nn.Module):
    """Parameterized weighted average of a sequence of vectors.
    This class uses an MLP to project each vector to a scalar, and then 
    normalize the resuling sequence of scalars to sum up to 1.
    """
    def __init__(self, size_in, size=None):
        super(WeightedAverage, self).__init__()
        self.activation = torch.tanh
        self.size_in = size_in
        self.size = size_in if size is None else size
        self.W1 = nn.Linear(self.size_in, self.size)
        self.W2 = nn.Linear(self.size, 1)

    def forward(self, h):
        alpha = F.softmax(self.W2(self.activation(self.W1(h))), dim=1)
        return (alpha.expand_as(h) * h).sum(dim=1)


def stringsim(a, b):
    """Levenshtein edit distance normalized by length of longer string."""
    return 1 - L.distance(a, b) / max(len(a), len(b))

def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return U_norm @ V_norm.t()

def cosine_matrix_batch(U, V): 
     "Returns the matrix of cosine similarity between each dim row of U[i] and each row of V[i], batchwise" 
     U_norm = U / U.norm(2, dim=2, keepdim=True) 
     V_norm = V / V.norm(2, dim=2, keepdim=True) 
     return U_norm @ V_norm.permute(0, 2, 1) 

def pearson_r(x, y, dim=0, eps=1e-8):
    "Returns Pearson's correlation coefficient."
    x1 = x - torch.mean(x, dim)
    x2 = y - torch.mean(y, dim)
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)

def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = torch.ones_like(x.data)
    return x[torch.triu(ones, diagonal=1) == 1]

def pairwise_concat(u, v):
    # (torch.diag(torch.cat([two, y], dim=1)[0]) @ torch.diag(torch.cat([x, two], dim=1)[0])).diag()
    v_ones = torch.cat([torch.ones_like(v), v], dim=1)
    u_ones = torch.cat([u, torch.ones_like(u)], dim=1)
    # make these guys into diagonal matrices now
    # and run bmm on them somehow
    
