import numpy as np
from sklearn.model_selection import train_test_split

from TP1.side_functions import get_accuracy, softmax, get_grads, get_loss, update_moment, compute_bias


def mini_batch_sgd(X: np.array, y: np.array, minibatch_size: int, lr: int, epochs, adam=False) -> int:
    """
    :param X: Inputs matrix of size NXL
    :param y: output matrix of size NXk
    :param minibatch_size: an interger to represent the size of the mini batch
    :param lr: an integer to give a certain learning rate
    :param adam: booléan to
    :return accuracy_on_unseen_data: The accuracy
    """
    y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
    y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot,
                                                        test_size=0.3, random_state=42)

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                  test_size=0.5, random_state=42)
    W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL
    # b = np.random.normal(0, 0.01, (len(np.unique(y)), 1))  # Biais of shape Kx1

    # W = np.concatenate((W, b), axis=1)

    best_W = None
    best_accuracy = 0
    nb_epochs = 50
    losses = []
    losses_validation = []
    accuracies = []

    # For ADAM_________________________________________________________________________________
    t= 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    m = np.zeros((len(np.unique(y)), X.shape[1]))
    v = np.zeros(m.shape)

    # _________________________________________________________________________________________
    X_train_t, y_train_t = X_train.T, y_train.T
    for epoch in range(epochs):
        loss = 0
        loss_validation = 0
       # np.random.seed(seed[epoch])
        shuffle = np.random.permutation(X_train_t.shape[1])
        X_train_t = X_train_t[:, shuffle]
        y_train_t = y_train_t[:, shuffle]

        for i in range(0, X_train_t.shape[0], minibatch_size):
            x = X_train_t[:, i:i+minibatch_size]
            y = y_train_t[:, i:i+minibatch_size]
            examples = x.shape[1]
            grad = np.zeros((W.shape[0], W.shape[1]))
            for example in range(examples):
                y_pred = softmax(W.dot(x[:, example]))
                grad -= get_grads(y[:, example], y_pred, x[:, example])
            grad /= minibatch_size
            if adam is True:
                t += 1
                m, v = update_moment(m, v, grad, beta_1, beta_2)
                hat_m, hat_v = compute_bias(m, v, t, beta_1, beta_2)
                W -= lr * hat_m / (np.sqrt(hat_v) + epsilon)
            else:
                W -= lr * grad

        # compute the loss on the train set
        for example in range(examples):
            y_pred = softmax(np.dot(W, X_train_t[:, example]))
            loss += get_loss(y_train_t[:, example], y_pred)

        loss_avg = loss / examples  # we take the average for the loss on the train set
        losses.append(loss_avg)

        nb_examples = X_validation.shape[0]

        for example in range(nb_examples):
            y_pred = softmax(np.dot(W, X_validation[example]))
            loss_validation += get_loss(y_validation[example], y_pred)
        losses_validation.append(loss_validation / nb_examples)

        # compute the accuracy on the validation set
        accuracy = get_accuracy(X_validation, y_validation, W)
        accuracies.append(accuracy)

        # select best parameters based ont the validation set
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W = W

    accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
    return losses, losses_validation, accuracies, accuracy_on_unseen_data, best_W
