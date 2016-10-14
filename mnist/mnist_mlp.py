import timeit
import os
import theano
import theano.tensor as T
import numpy as np

from mnist.mlp import MLP
from lib2 import load_data


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):

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

    rng = np.random.RandomState(1234)

    classifier = MLP(
        rng=rng, input_=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out
    )

    cost = (
        classifier.negative_log_likelihood(y) +
        L1_reg * classifier.L1 +
        L2_reg * classifier.L2
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    grad_params = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * grad_params)
        for param, grad_params in zip(classifier.params, grad_params)
    ]

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

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_freq = min(n_train_batches, patience // 2)
    best_valid_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_freq == 0:
                valid_losses = [valid_model(i)
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
                    best_iter = iter

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        'epoch %i | minibatch %i/%i | test error %f' %
                        (
                            epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f obtained at'
          ' iteration %i, with performance of %f' %
          (best_valid_loss * 100., best_iter + 1, test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' %
          (epoch, 1. * epoch / (end_time - start_time)))
    print('The code for file ' + os.path.split(__file__)[
        1] + ' ran for %.1fs'
          % (end_time - start_time))
