import os
import timeit

import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T

from lib2 import load_data
from mnist.logistic_regression import LogisticRegression


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz', batch_size=600):
    dataset = load_data(dataset)
    n_in = 28 * 28
    n_out = 10

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('Building model...')

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input_=x, n_in=n_in, n_out=n_out)
    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    grad_W = T.grad(cost=cost, wrt=classifier.W)
    grad_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * grad_W),
               (classifier.b, classifier.b - learning_rate * grad_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print("Training the model")
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_freq = min(n_train_batches, patience // 2)
    best_valid_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_freq == 0:
                valid_losses = [validate_model(i)
                                for i in range(n_valid_batches)]
                this_valid_loss = np.mean(valid_losses)

                print(
                    'epoch %i | minibatch %i/%i | validation error %f' %
                    (
                        epoch, minibatch_index + 1, n_train_batches,
                        this_valid_loss * 100.
                    )
                )
                if this_valid_loss < best_valid_loss:
                    if this_valid_loss < best_valid_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                best_valid_loss = this_valid_loss

                test_losses = [test_model(i) for i in range(n_test_batches)]
                test_score = np.mean(test_losses)

                print(
                    'epoch %i | minibatch %i/%i | test error %f' %
                    (
                        epoch, minibatch_index + 1, n_train_batches,
                        test_score * 100.
                    )
                )

                with open('mnist_model.pkl', 'wb') as f:
                    pickle.dump(classifier, f)

        if patience <= iter:
            done_looping = True
            break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f with '
          'performance of %f' % (best_valid_loss * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' %
          (epoch, 1. * epoch / (end_time - start_time)))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs'
          % (end_time - start_time))


def predict():
    classifier = pickle.load(open('mnist_mlp_model.pkl'))

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print('Predicted values for the first 10 examples in test set:')
    print(predicted_values)
