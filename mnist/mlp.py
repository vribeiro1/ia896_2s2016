import theano.tensor as T

from mnist.hidden_layer import HiddenLayer
from mnist.logistic_regression import LogisticRegression


class MLP:
    def __init__(self, rng, input_, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(
            rng=rng, input_=input_, n_in=n_in, n_out=n_hidden, activation=T.tanh
        )

        self.log_reg_layer = LogisticRegression(
            input_=self.hidden_layer.output, n_in=n_hidden, n_out=n_out
        )

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_reg_layer.W).sum()
        self.L2 = (self.hidden_layer.W ** 2).sum() + (self.log_reg_layer.W ** 2).sum()

        self.negative_log_likelihood = self.log_reg_layer.negative_log_likelihood
        self.errors = self.log_reg_layer.errors
        self.params = self.hidden_layer.params + self.log_reg_layer.params
        self.input_ = input_
