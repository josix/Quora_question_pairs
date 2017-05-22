import numpy as np

def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    product = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return product/(v1_norm*v2_norm)*1.0

def jaccard(v1, v2):
    numerator = 0
    denominator = 0
    for x,y in zip(v1, v2):
        numerator += min(x, y)
        denominator += max(x, y)
    return numerator*1.0/denominator

if __name__ == "__main__":
    pass
