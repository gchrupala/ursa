from ursa.kernel import Kernel
import ursa.util as U
from nltk.tree import Tree
from hypothesis.strategies import builds, text, recursive, lists, booleans
from hypothesis import given
from string import ascii_letters
import numpy as np

examples = [ "(NP (D a) (N dog))",
             "(NP (D the) (N cat))",
             "(VP (NP (D the) (N cat)) (V sleeps))"]
trees = [Tree.fromstring(e) for e in examples ]

tree =recursive(text(ascii_letters), 
                lambda children:  builds(Tree, text(ascii_letters), 
                                         lists(elements=children))).filter(lambda t: not isinstance(t, str))
@given(t1=tree, t2=tree)
def test_symmetric(t1, t2):
    K = Kernel()
    assert K(t1, t2) == K(t2, t1)

def test_NP_D_N():
    K = Kernel()
    assert K(trees[0], trees[1]) == 1.0

def test_self_NP():
    K = Kernel()
    assert K(trees[0], trees[0]) == 6.0
    assert K(trees[0], trees[1]) < 6.0    
    K = Kernel(ignore_terminals=True)
    assert K(trees[0], trees[1]) == 6.0

def test_ignore_terminals():
    K = Kernel(ignore_terminals=True)
    assert K(trees[0][1], trees[1][1]) == 1.0
    K = Kernel(ignore_terminals=False)
    assert K(trees[0][1], trees[1][1]) == 0.0

@given(t1=tree, t2=tree)    
def test_alpha(t1, t2):
    k_half = Kernel(alpha=0.5)(t1, t2)
    k_one  = Kernel(alpha=1.0)(t1, t2)
    assert k_half == 0.0 and k_one == 0.0 or k_half < k_one

@given(t1=tree, t2=tree)
def test_normalize(t1, t2):
    K = Kernel()
    k = K(t1, t2) / (K(t1, t1) * K(t2, t2))**0.5
    assert k >= 0.0 and k <= 1.0

@given(t1=tree, t2=tree)
def test_ftk(t1, t2):
    K = Kernel()
    n_ftk = K.ftk(K.nodemap(t1), K.nodemap(t2))
    n_naive = K(t1, t2)
    assert n_ftk == n_naive
    
@given(trees=lists(elements=tree), normalize=booleans())
def test_pairwise_ftk(trees, normalize):
    K = Kernel()
    M_naive = U.pairwise(K, trees, parallel=False, normalize=normalize)
    M_ftk   = K.pairwise(trees, normalize=normalize)
    assert np.allclose(M_naive, M_ftk)
    
