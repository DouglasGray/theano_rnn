import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import data_loader as loader
import matplotlib.pyplot as plt
from datetime import datetime

# Network variables
input_dim = 60
hidden_dim = 50
output_dim = 30

# Misc
lr = 0.01
trans_cost = 0.01
size_mult = 10
lmbda = 0.01

W1 = theano.shared(rng.uniform(-np.sqrt(1. / (hidden_dim + input_dim)), np.sqrt(1. / (hidden_dim + input_dim)),
                               (3, hidden_dim, input_dim + hidden_dim)), name='W1')
W2 = theano.shared(rng.uniform(-np.sqrt(1. / (hidden_dim + output_dim)), np.sqrt(1. / (hidden_dim + output_dim)),
                               (3, output_dim, hidden_dim + output_dim)), name='W2')

b1 = theano.shared(np.zeros((3, hidden_dim)), name='b1')
b2 = theano.shared(np.zeros((3, output_dim)), name='b2')


X = T.dmatrix('X')
r = T.dmatrix('r')


def calculate_position(x_in, s_t1_prev, s_t2_prev):
    x_t1 = T.concatenate([s_t1_prev, x_in])
    z_t1 = T.nnet.sigmoid(W1[0].dot(x_t1) + b1[0])
    r_t1 = T.nnet.sigmoid(W1[1].dot(x_t1) + b1[1])

    h_t1 = T.concatenate([s_t1_prev * r_t1, x_in])
    c_t1 = T.tanh(W1[2].dot(h_t1) + b1[2])
    s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

    x_t2 = T.concatenate([s_t2_prev, s_t1])
    z_t2 = T.nnet.sigmoid(W2[0].dot(x_t2) + b2[0])
    r_t2 = T.nnet.sigmoid(W2[1].dot(x_t2) + b2[1])

    h_t2 = T.concatenate([s_t2_prev * r_t2, s_t1])
    c_t2 = T.tanh(W2[2].dot(h_t2) + b2[2])
    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

    return s_t1, s_t2


def calculate_pnl(qty, qty_prev, r_t):
    return T.sum(size_mult * (qty_prev * r_t - trans_cost * T.abs_(qty - qty_prev)))


print('Creating network graph...')

[_, positions], pos_updates = theano.scan(calculate_position,
                                          sequences=X,
                                          outputs_info=[T.zeros(hidden_dim), T.zeros(output_dim)])

pnl, pnl_updates = theano.scan(calculate_pnl, sequences=[positions[1:, :], positions[:-1, :], r])

sharpe = T.mean(pnl) / (T.std(pnl) + 1e-6) - lmbda * (W1[0].norm(2) + W1[1].norm(2) + W1[2].norm(2) +
                                                     W2[0].norm(2) + W2[1].norm(2) + W2[2].norm(2))


dW1 = T.grad(sharpe, W1)
db1 = T.grad(sharpe, b1)
dW2 = T.grad(sharpe, W2)
db2 = T.grad(sharpe, b2)

train_network = theano.function(inputs=[X, r],
                                outputs=[sharpe, positions, pnl],
                                updates=[(W1, W1 + lr * dW1),
                                         (W2, W2 + lr * dW2),
                                         (b1, b1 + lr * db1),
                                         (b2, b2 + lr * db2)])

test_network = theano.function(inputs=[X, r],
                               outputs=[sharpe, positions, pnl])


def train(train_data, val_data, max_epochs_no_improve, max_epochs):
    no_improve = 0
    best_sr = -np.inf
    epochs = 0
    while no_improve < max_epochs_no_improve and epochs < max_epochs:
        train_network(train_data['inputs'], train_data['targets'])
        sr, _, _ = test_network(val_data['inputs'], val_data['targets'])

        print("Cost on validation set after epoch {}: {}".format(epochs, sr))

        if sr > best_sr:
            best_sr = sr
            no_improve = 0
            loader.save_params(W1, 'W1.npy')
            loader.save_params(W2, 'W2.npy')
            loader.save_params(b1, 'b1.npy')
            loader.save_params(b2, 'b2.npy')
        else:
            no_improve += 1
        epochs += 1


def test(test_data):
    sr, pos, loss = test_network(test_data['inputs'], test_data['targets'])

    print("Cost on test set: {}".format(sr))

    plt.plot(np.cumsum(loss))
    plt.title('pnl')
    plt.show()

    for ix in range(output_dim):
        plt.plot(size_mult * pos[:, ix])
    plt.title('positions (all stocks)')
    plt.show()


start = datetime(2010, 1, 1)
end = datetime(2017, 3, 1)

train_X, val_X, test_X = loader.load_data_from_yahoo(start, end, test_split=0.3, val_split=0.2)

train(train_X, val_X, max_epochs_no_improve=50, max_epochs=200)

W1 = theano.shared(loader.load_params('W1.npy'), name='W1')
W2 = theano.shared(loader.load_params('W2.npy'), name='W2')
b1 = theano.shared(loader.load_params('b1.npy'), name='b1')
b2 = theano.shared(loader.load_params('b2.npy'), name='b2')

test(test_X)