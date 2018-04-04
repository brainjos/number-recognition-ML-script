from mnist import MNIST
mndata = MNIST('./data')
mndata.gz = True
images, labels = mndata.load_training()

for x in range(5):
    print(images[x])
    print()
