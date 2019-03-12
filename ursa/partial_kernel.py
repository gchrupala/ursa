from itertools import combinations


def subsequences(z):
    z = list(z)
    for r in range(1, len(z)+1): 
        yield from combinations(z, r)
        
