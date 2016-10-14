import sys

from mnist.mnist import sgd_optimization_mnist
from mnist.mnist_mlp import test_mlp


def main():
    mnist_models = {
        'mnist': sgd_optimization_mnist,
        'mnist_mlp': test_mlp
    }

    if len(sys.argv) != 2:
        print( """
        correct usage is 'python mnist <mode>' which mode can be
        'mnist' or 'mnist_mlp'
        """
        )
        exit(0)

    mnist_model = mnist_models[sys.argv[1]]
    mnist_model()

if __name__ == '__main__':
    main()
