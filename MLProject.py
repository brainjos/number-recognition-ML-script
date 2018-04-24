from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import numpy as np
import cv2
import math
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


#placeholder value for input. (None implies that dimension can be of any length)
x = tf.placeholder(tf.float32, [None, 784])

#Variables: modifiable tensors that can be used and modified by the computation.
#weights initialized to zero
W = tf.Variable(tf.zeros([784, 10]))
#bias initialized to zero
b = tf.Variable(tf.zeros([10]))

#complete model:
y = tf.nn.softmax(tf.matmul(x, W) + b)

#traning setup:
#placeholder for inputting correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indicies=[1]))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#lauch interactive session and initialize varaibles we created
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#run training step 1000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#evaluating model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#OUR IMAGES:
images = np.zeros((100,784))
correct_vals = np.zeros((100,10))

for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    i = 0
    for no in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: #2395
        gray = cv2.imread("./img/preprocessed/trial" + str(t) + "_pre_" + str(no) + ".png", 0)
        gray = cv2.resize(255-gray, (28, 28))
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        rows,cols = gray.shape

        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted        

        cv2.imwrite("./img/processed/trial" + str(t) + "_" + str(no) + ".png", gray)

        flatten = gray.flatten() / 255.0

        images[(t-1)*10 + i] = flatten
        correct_val = np.zeros((10))
        correct_val[no] = 1
        correct_vals[(t-1)*10 + i] = correct_val
        i += 1

prediction = tf.argmax(y,1)

print(sess.run(prediction, feed_dict={x: images, y_: correct_vals}))
print(sess.run(accuracy, feed_dict={x: images, y_: correct_vals}))
