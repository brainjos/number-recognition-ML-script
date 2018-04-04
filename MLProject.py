import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
mndata = MNIST('./data')
mndata.gz = True
images, labels = mndata.load_training()
from sklearn.datasets import load_digits
digits = load_digits()

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(images[0:5], labels[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()
