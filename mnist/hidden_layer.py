import theano
import numpy as np
import theano.tensor as T


class HiddenLayer:
    def __init__(self, rng, input_, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input_

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

        lin_output = T.dot(input_, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)
        self.params = [self.W, self.b]
