import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=20, centers=5, random_state=42)

# Računanje poveznica između klastera pomoću metode 'ward'
Z = linkage(X, method='single')


# Prikazivanje dendrograma
plt.figure(figsize=(10, 5))
dendrogram(Z)

plt.show()