import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Učitavanje slike
image = mpimg.imread('example.png')

# Izdvajanje dimenzija slike
w, h, d = tuple(image.shape)

# Prebacivanje slike u jednodimenzionalni niz
image_array = np.reshape(image, (w * h, d))

# Primjena K-means algoritma na niz piksela
n_clusters = 10  # broj klastera za kvantizaciju boje
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(image_array)

# Kvantizacija boje slike
compressed_image_array = kmeans.predict(image_array)
compressed_image = np.reshape(compressed_image_array, (w, h))

# Izračun kompresije
original_size = w * h * d
compressed_size = n_clusters * (d + 1)  # broj klastera * (broj dimenzija + 1)
compression_ratio = original_size / compressed_size

# Prikaz slika
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Originalna slika')
ax2.imshow(compressed_image, cmap='gray')
ax2.set_title('Kvantizirana slika s {} klastera\nKompresijski omjer: {:.2f}'.format(n_clusters, compression_ratio))
plt.show()