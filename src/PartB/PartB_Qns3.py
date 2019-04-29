import tensorflow as tf
import numpy as np
import pylab as plt
import math
from sklearn.model_selection import KFold

# some global variables
num_features = 8
epochs = 500
seed = 10
batch_size = 32
beta = 10**-3
learning_rate = 10**-7
no_exps = 5

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(seed)
np.random.seed(seed)

#scale the training input features
def scale(inputs):
    return (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)

def train(X, Y, units):

    #we do k fold split from the 70% training data
    X_train_sets = []
    X_test_sets = []
    Y_train_sets = []
    Y_test_sets = []



    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):

        X_train_sets.append(X[train_index])
        X_test_sets.append(X[test_index])
        Y_train_sets.append(Y[train_index])
        Y_test_sets.append(Y[test_index])

    mean_err = []
    for hidden_units in units:

        print('Evaluating Hidden Units %g' % hidden_units)
        print('===========================')
        # Create the model
        x = tf.placeholder(tf.float32, [None, num_features])
        y_ = tf.placeholder(tf.float32, [None, 1])

        # Built the graph for the deep net

        w1 = tf.Variable(tf.truncated_normal([num_features, hidden_units],
                                             stddev=1.0 / math.sqrt(float(num_features))),
                         name='weights')

        b1 = tf.Variable(tf.zeros([hidden_units]), name='biases')

        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        w2 = tf.Variable(tf.truncated_normal([hidden_units, 1],
                                             stddev=1.0 / np.sqrt(hidden_units),
                                             dtype=tf.float32), name='weights')

        b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')

        y = tf.matmul(h1, w2) + b2

        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

        loss = tf.reduce_mean(tf.square(y_ - y) + beta * regularization)
        error = tf.reduce_mean(tf.square(y_ - y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

        err =[]
        for i in range(5):

            print('%d FOLD' % (i+1))
            trainX = X_train_sets[i]
            trainY = Y_train_sets[i]
            testX = X_test_sets[i]
            testY = Y_test_sets[i]

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                N = len(trainX)
                idx = np.arange(N)

                for j in range(epochs):

                    np.random.shuffle(idx)
                    trainX = trainX[idx]
                    trainY = trainY[idx]

                    for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                        train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

                err.append(error.eval(feed_dict={x: testX, y_: testY}))
                print('Validation error is %g' % err[i])

        mean_err.append(np.mean(err))
    optimal = units[np.argmin(mean_err)]
    print()
    print('Optimal Hidden Units is %g' % optimal)

    return optimal, mean_err


def main():

    #we need to vary the learning rate according to this search space

    units = [20,40,60,80,100]

    # read and divide data into test and train sets
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()

    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

    m = 3 * X_data.shape[0] // 10
    trainX, trainY = X_data[m:], Y_data[m:]

    testX, testY = X_data[0:m], Y_data[0:m]

    # scale the input features
    trainX = scale(trainX)
    testX = scale(testX)

    n = trainX.shape[0]
    optimal_neurons=[]
    cv_errors = []
    for i in range(no_exps):
        print('Experiment number %d' % (i + 1))
        idx2 = np.arange(n)
        np.random.shuffle(idx2)
        optimal_n,cv_e= train(trainX[idx2], trainY[idx2],units)
        optimal_neurons.append(optimal_n)
        cv_errors.append(cv_e)

    hidden_counts = np.zeros(len(units))
    for exp in range(no_exps):
        hidden_counts[optimal_neurons[exp] // 20 - 1] += 1

    final_optimal = units[np.argmax(hidden_counts)]
    print('After %d experiments %d is the optimal number of hidden neurons' % (no_exps, final_optimal))

    plt.figure(1)
    plt.plot(range(no_exps), optimal_neurons, marker='x', linestyle='None')
    plt.yticks(units)
    plt.xticks(range(no_exps), np.arange(no_exps) + 1)
    plt.xlabel('experiment')
    plt.ylabel('optimum number of hidden units')
    plt.savefig('./figures/PARTB_QNS3_a.png')


    optimal_errors = test_optimal(trainX,trainY,testX,testY,final_optimal)

    plt.figure(2)
    plt.plot(range(epochs), optimal_errors)
    plt.xlabel('Epochs')
    plt.ylabel('Test Errors')
    plt.title('Test Errors Vs Epochs')
    plt.savefig('./figures/PARTB_QNS3_b.png')

    i = 3
    for cv_error in cv_errors:
        plt.figure(i)
        plt.plot(units, cv_error, 'x')
        plt.xlabel('Hidden Units')
        plt.ylabel('Mean CV errors')
        plt.title('Mean CV errors VS Hidden Units EXP %d' %(i-2))
        plt.savefig('./figures/PARTB_QNS3_c_EXP_%d.png' % (i-2))
        i+=1

    plt.show()

def test_optimal(X,Y,pureTestX,pureTestY,optimal):
    # for the optimal learning rate
    x = tf.placeholder(tf.float32, [None, num_features])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Built the graph for the deep net

    w1 = tf.Variable(tf.truncated_normal([num_features, optimal],
                                         stddev=1.0 / math.sqrt(float(num_features))),
                     name='weights')

    b1 = tf.Variable(tf.zeros([optimal]), name='biases')

    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([optimal, 1],
                                         stddev=1.0 / np.sqrt(optimal),
                                         dtype=tf.float32), name='weights')

    b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')

    y = tf.matmul(h1, w2) + b2

    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

    loss = tf.reduce_mean(tf.square(y_ - y) + beta * regularization)
    error = tf.reduce_mean(tf.square(y_ - y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print()
        print('Now evaluating on optimal learning rate')
        N = len(X)
        idx = np.arange(N)
        optimal_err = []
        for j in range(epochs):

            np.random.shuffle(idx)
            trainX = X[idx]
            trainY = Y[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            optimal_err.append(error.eval(feed_dict={x: pureTestX, y_: pureTestY}))

            if j % 100 == 0:
                print('iter %d: test error %g' % (j, optimal_err[j]))

    return optimal_err



if __name__ == '__main__':
  main()