from functools import reduce
from typing import Sequence,  Union, List, Iterable
from typing_extensions import Protocol

class Labeled(Protocol):
    def label(self) -> str: ...

class TreeLike(Sequence['TreeLike'], Labeled): ...

# These functions work with nltk.tree.Tree

def label(n: Union[Labeled, str]) -> str:
    if isinstance(n, str):
        return n
    else:
        return n.label()

def leaf(n: TreeLike) -> bool:
    return isinstance(n, str)

def children(n: TreeLike) -> Sequence[TreeLike]:
    if isinstance(n, str):
        return []
    else:
        return n[:]

class Kernel:
    """Class to hold configuration of the kernel function."""
    def __init__(self, label=label, leaf=leaf, children=children, alpha: float=1.0, ignore_terminals: bool=False):
        self.label = label
        self.leaf = leaf
        self.children = children
        self.alpha = alpha
        self.ignore_terminals = ignore_terminals

    def subtrees(self, t: TreeLike) -> Iterable[TreeLike]:
        """Yields all subtrees of a tree t."""
        if leaf(t):
            pass
        else:
            yield t
            for c in self.children(t):
                yield from self.subtrees(c)

    def preterm(self, t: TreeLike) -> bool:
        """Returns True if node t is a pre-terminal."""
        return len(self.children(t)) == 1 and self.leaf(self.children(t)[0])
    
    def production(self, t: TreeLike) -> List[str]:
        """Returns the productiona at node t, i.e. the list of children's labels."""
        return [ self.label(c) for c in self.children(t) ]

    def C(self, n1: TreeLike, n2: TreeLike) -> float:
        # both nodes are preterminals and have same productions
        if self.preterm(n1) and self.preterm(n2) and self.label(n1) == self.label(n2) and (self.production(n1) == self.production(n2) or self.ignore_terminals):
            return self.alpha
        # both nodes are non-terminals and have same productions
        elif not self.preterm(n1) and not self.preterm(n2) and self.label(n1) == self.label(n2) and self.production(n1) == self.production(n2):
            return self.alpha * product(1 + self.C(self.children(n1)[i], self.children(n2)[i]) for i in range(len(self.children(n1))))
        else:
            return 0

    def __call__(self, t1: TreeLike, t2: TreeLike):
        """Returns the number of shared tree fragments between trees t1 and t2, discounted by alpha."""
        N = sum(self.C(n1, n2) for n1 in self.subtrees(t1) for n2 in self.subtrees(t2))
        return N

def product(xs: Iterable[float]) -> float:
    return reduce(lambda a, b: a*b, xs, 1.0)

