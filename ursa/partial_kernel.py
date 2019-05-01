from itertools import combinations


def subsequences(z):
    z = list(z)
    for r in range(1, len(z)+1): 
        yield from combinations(z, r)
        
def C(self, n1: TreeLike, n2: TreeLike) -> float:
    # both nodes are preterminals and have same productions
    if self.preterm(n1) and self.preterm(n2) and self.label(n1) == self.label(n2) and (self.production(n1) == self.production(n2)):
        return self.alpha
    # both nodes are non-terminals
    # generate all possible subproductions and sum
    elif not self.preterm(n1) and not self.preterm(n2) and self.label(n1) == self.label(n2) and self.production(n1) == self.production(n2):
        return self.alpha * product(1 + self.C(self.children(n1)[i], self.children(n2)[i]) for i in range(len(self.children(n1))))
    else:
        return 0
