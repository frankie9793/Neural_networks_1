import tensorflow as tf
import numpy as np
import pylab as plt
import math
from sklearn.model_selection import KFold

# some global variables
num_features = 8
epochs = 500
seed = 10
learning_rate = 10 ** -9
hidden_units = 20
batch_size = 32
beta = 10 ** -3

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(seed)
np.random.seed(seed)


# scale the training input features
def scale(inputs):
    return (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)

def train(X, Y, puretestX, puretestY, prob):
    # we do k fold split from the 70% training data
    X_train_sets = []
    X_test_sets = []
    Y_train_sets = []
    Y_test_sets = []
    error_sets_nodrop = []

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train_sets.append(X[train_index])
        X_test_sets.append(X[test_index])
        Y_train_sets.append(Y[train_index])
        Y_test_sets.append(Y[test_index])

    # after the for-loop above we have 5 lists inside the list of sets

    # Create the model
    x = tf.placeholder(tf.float32, [None, num_features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    keep_prob = tf.placeholder(tf.float32)

    # Built the graph for the deep net

    w1 = tf.Variable(tf.truncated_normal([num_features, hidden_units],
                                         stddev=1.0 / math.sqrt(float(num_features))),
                     name='weights')

    b1 = tf.Variable(tf.zeros([hidden_units]), name='biases')

    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    h1_dropout = tf.nn.dropout(h1, keep_prob)

    w2 = tf.Variable(tf.truncated_normal([hidden_units, 1],
                                         stddev=1.0 / np.sqrt(hidden_units),
                                         dtype=tf.float32), name='weights')

    b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')

    y = tf.matmul(h1_dropout, w2) + b2

    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

    loss = tf.reduce_mean(tf.square(y_ - y) + beta * regularization)
    error = tf.reduce_mean(tf.square(y_ - y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    for i in range(5):
        trainX = X_train_sets[i]
        trainY = Y_train_sets[i]
        testX = X_test_sets[i]
        testY = Y_test_sets[i]
        test_err_nodrop = []

        print('Evaluating %g fold' % (i + 1))
        print('========================')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            N = len(trainX)
            idx = np.arange(N)

            for i in range(epochs):

                np.random.shuffle(idx)
                trainX = trainX[idx]
                trainY = trainY[idx]

                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    # _, error_ = sess.run([train_op, error],feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                    train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: prob})

                err_nodrop = error.eval(feed_dict={x: puretestX, y_: puretestY, keep_prob: 1.0})

                test_err_nodrop.append(err_nodrop)

                if i % 100 == 0:
                    print('iter %d: test error is %g' % (i, test_err_nodrop[i]))

        error_sets_nodrop.append(test_err_nodrop)

            # this is the error set appended for 1 fold across 500 epochs
    return error_sets_nodrop

def main():
    # read and divide data into test and train sets

    keep_probs = [0.9, 1.0]

    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()

    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

    m = 3 * X_data.shape[0] // 10
    trainX, trainY = X_data[m:], Y_data[m:]

    # TEST IS PURELY FOR TESTING WE DO NOT TRAIN THESE
    testX, testY = X_data[0:m], Y_data[0:m]

    # scale the input features
    trainX = scale(trainX)
    testX = scale(testX)

    mean_errors = []

    for keep_prob in keep_probs:
        print("Evaluation with keep probability of %g" % keep_prob)
        errors_nodrops = train(trainX, trainY, testX, testY, keep_prob)
        mean = np.mean(np.array(errors_nodrops), axis=0)
        mean_errors.append(mean)

    plt.figure(1)
    plt.plot(range(epochs), mean_errors[0],label='prob_0.9')
    plt.plot(range(epochs), mean_errors[1],label='prob_1.0')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Test Errors')
    plt.legend()
    plt.title('Mean Test Errors by dropping weights')
    plt.savefig('./figures/PARTB_QNS4_3Layer.png')

    plt.show()


if __name__ == '__main__':
    main()