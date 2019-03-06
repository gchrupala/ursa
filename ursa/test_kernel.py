from ursa.kernel import Kernel
from nltk.tree import Tree

examples = [ "(NP (D a) (N dog))",
             "(NP (D the) (N cat))",
             "(VP (NP (D the) (N cat)) (V sleeps))"]
trees = [Tree.fromstring(e) for e in examples ]

def test_symmetric():
    K = Kernel()
    for t1 in trees:
        for t2 in trees:
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

def test_alpha():
    K = Kernel()
    assert K(trees[0], trees[1], alpha=0.5) < K(trees[0], trees[1], alpha=1.0)

def test_normalize():
    K = Kernel()
    for t1 in trees:
        for t2 in trees:
            k = K(t1, t2) / (K(t1, t1) * K(t2, t2))**0.5
            assert k >= 0.0 and k <= 1.0
