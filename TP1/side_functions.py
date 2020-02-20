import numpy as np


def softmax(x: np.ndarray) -> np.array:
    """
    :param x: Takes in the training data wich is a vector of 1XL
    :return: Returns normalized probability distribution consisting of probabilities proportional to the exponentials
            of the input numbers. Vector of size
    """
    e_x = np.exp(x - max(x))
    return e_x / e_x.sum(0)


def get_accuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> int:
    """
    Definition of accuracy: proportion of nb_examples for witch the model produces the correct output.
    :param X: Matrice of inputs
    :param y: Matrice of outputs
    :param W: matrcie of Weight
    :return:
    """

    good_guess = 0
    bad_guess = 0

    for img, correct in zip(X, y):

        y_pred = softmax(W.dot(img))
        pred = np.argmax(y_pred)

        if (correct[pred] == 1):
            good_guess += 1
        else:
            bad_guess += 1

    # print('#good = {}'.format(good_guess))
    # print('#bad = {}'.format(bad_guess))
    # print("................")

    return good_guess / (bad_guess + good_guess) * 100


def get_grads(y: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> int:
    """
    Inner product
    :param y:
    :param y_pred: softmax de x is a matrix
    :param X:
    :return:
    """
    delta = y - y_pred
    return np.outer(delta, X)


def get_loss(y: np.ndarray, y_pred: np.ndarray) -> int:
    """
    :param y: Is a vector of dimension 1XK
    :param y_pred: Is a vector of dimension 1XK
    :return: The loss witch is an int
    """
    return -np.dot(y, np.log(y_pred))


def update_moment(m, v, grad, beta_1, beta_2):
    m = beta_1 * m + (1 - beta_1) * grad
    v = beta_2 * v + (1 - beta_2) * grad ** 2
    return m, v


def compute_bias(m, v, t, beta_1, beta_2):
    hat_m = m / (1 - beta_1 ** t)
    hat_v = v / (1 - beta_2 ** t)
    return  hat_m, hat_v


def generate_minibatchs(X_train: np.array, y_train, minibatch_size: int) -> [range, np.array, np.array, int]:
    batches = range(0, X_train.shape[0], minibatch_size)
    nb_of_batch = X_train.shape[0] // minibatch_size
    nb_examples = X_train.shape[0] // nb_of_batch
    X_train = [X_train[i:i + minibatch_size, :] for i in batches]
    y_train = [y_train[i:i + minibatch_size, :] for i in batches]
    return batches, X_train, y_train, nb_examples
