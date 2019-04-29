import tensorflow as tf
import numpy as np
import pylab as plt
import math
import time

# some global variables
num_classes = 6
num_features = 36
epochs = 1000
seed = 10
learning_rate = 0.01
batch_size = 8
beta = 0.000001


#Sets the threshold for what messages will be logged ---> ?
tf.logging.set_verbosity(tf.logging.ERROR) #tf.logging.ERROR == 40.
tf.set_random_seed(seed)
np.random.seed(seed)

def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)


def train(hidden_units):  # we can vary the para for later parts of the question here --->

    # Read train data
    train_input = np.loadtxt('sat_train.txt', delimiter=' ')
    trainX, train_Y = train_input[:, :36], train_input[:, -1].astype(int)
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
    train_Y[train_Y == 7] = 6

    trainY = np.zeros((train_Y.shape[0], num_classes))
    trainY[np.arange(train_Y.shape[0]), train_Y - 1] = 1  # one hot matrix

    # experiment with small datasets ---> we can vary this based on the accuracy and time taken to run
    trainX = trainX[:4435]
    trainY = trainY[:4435]

    #Read test data
    test_input = np.loadtxt('sat_test.txt', delimiter=' ')
    testX, test_Y = test_input[:, :36], test_input[:, -1].astype(int)
    testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
    test_Y[test_Y == 7] = 6

    testY = np.zeros((test_Y.shape[0], num_classes))
    testY[np.arange(test_Y.shape[0]), test_Y - 1] = 1  # one hot matrix


    # Create the model
    x = tf.placeholder(tf.float32, [None, num_features])
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Build the graph for the deep net

    # do a truncated normal for the weights for the hidden layer
    w1 = tf.Variable(tf.truncated_normal([num_features, hidden_units],
                                         stddev=1.0 / math.sqrt(float(num_features))),
                     name='weights')
    # set biases to zeros
    b1 = tf.Variable(tf.zeros([hidden_units]), name='biases')

    # for perceptron layer we use sigmoid function
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    # do a truncated normal for the weights of the output layer
    w2 = tf.Variable(tf.truncated_normal([hidden_units, num_classes],
                                         stddev=1.0 / math.sqrt(float(hidden_units))),
                     name='weights')
    # set biases to 0
    b2 = tf.Variable(tf.zeros([num_classes]), name='biases')

    y = tf.matmul(h1, w2) + b2

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)

    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))

    loss = tf.reduce_mean(cross_entropy + beta * regularization)

    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        errors = []
        total_time = 0.0

        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]
            epoch_time = 0.0

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                start_time = time.time()
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                end_time = time.time()
                epoch_time += end_time - start_time


            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            errors.append(error.eval(feed_dict={x: trainX, y_: trainY}))
            total_time += epoch_time / float(N// batch_size)

            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('         training error %g' % (errors[i]))


        time_epoch = (total_time/float(epochs))*1000

    return test_acc,errors,time_epoch


def main():

    h_units = [5,10,15,20,25]
    accuracies =[]
    errors = []
    times = []


    for i in h_units:

        print()
        print('Evaluating hidden units size %d' % (i))
        print('==============================')
        accuracy_batch, error_batch, time_batch = train(i)
        accuracies.append(accuracy_batch)
        times.append(time_batch)
        errors.append(error_batch)


    print()
    print('At convergence')
    print('==============================')
    i = 0
    for units in h_units:
        print('No of Hidden Units %g' % units)
        print('====================')
        print('Test Accuracy is %g' % accuracies[i][epochs-1])
        print('Time taken for updating %g' % times[i])
        print()
        i += 1



    plt.figure(1)
    plt.plot(range(epochs), accuracies[0], label="units_5")
    plt.plot(range(epochs), accuracies[1], label="units_10")
    plt.plot(range(epochs), accuracies[2], label="units_15")
    plt.plot(range(epochs), accuracies[3], label="units_20")
    plt.plot(range(epochs), accuracies[4], label="units_25")
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.title('Test Accuracies of different hidden units')
    plt.savefig('./figures/accuracy_hunits.png')

    plt.figure(2)
    plt.plot(range(epochs), errors[0], label="units_5")
    plt.plot(range(epochs), errors[1], label="units_10")
    plt.plot(range(epochs), errors[2], label="units_15")
    plt.plot(range(epochs), errors[3], label="units_20")
    plt.plot(range(epochs), errors[4], label="units_25")
    plt.xlabel('epochs')
    plt.ylabel('training errors')
    plt.legend()
    plt.title('Training Errors of different hidden units')
    plt.savefig('./figures/errors_hunits.png')

    ind = np.arange(4)
    width = 0.35

    plt.figure(3)
    plt.plot(h_units,times)
    plt.plot(h_units, times,'o')
    plt.xlabel('Hidden_units')
    plt.ylabel('Average time 1 epoch')
    plt.xticks([5,10,15,20,25])
    plt.title('Time Vs Hidden_units')
    plt.savefig('./figures/time_hunits.png')

    plt.show()



if __name__ == '__main__':
  main()