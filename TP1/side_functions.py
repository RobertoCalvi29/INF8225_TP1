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

def get_gradient():
    pass


def update_moment(moment_dict:dict, gradient:int):
    beta_1 = None
    beta_2 = None
    pass


def compute_bias(moment_dict:dict):
    pass


def update_theta(theta , moment_dict):
    pass


# def generate_minibatch(X_train: np.array, y_train, minibatch_size: int) -> [range, np.array, np.array, int]:
#     """
#     Function to allow minibatch creation
#     :param theta: Matrix of size NXL+1
#     :param minibatch_size: what it says
#     :param
#     :return:
#     """
#     batches = range(0, X_train.shape[0], minibatch_size)
#     nb_of_batch = X_train.shape[0] // minibatch_size
#     nb_examples = X_train.shape[0] // nb_of_batch
#     X_train = [X_train[i:i + minibatch_size, :] for i in batches]
#     y_train = [y_train[i:i + minibatch_size, :] for i in batches]
#     return batches, X_train, y_train, nb_examples
