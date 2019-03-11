from typing import Any, Sequence
import conllu.models as C
from collections import OrderedDict


def label(n: C.TokenTree) -> str:
    return n.token['upostag']

def leaf(n: C.TokenTree) -> bool:
    return len(n.children) == 0

def children(n: C.TokenTree) -> Sequence[C.TokenTree]:
    return n.children


