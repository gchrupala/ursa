from typing import Any, Sequence
import conllu.models as C
from collections import OrderedDict


def label(n: C.TokenTree) -> str:
    return n.token['deprel']

def children(n: C.TokenTree) -> Sequence[C.TokenTree]:
    return n.children


