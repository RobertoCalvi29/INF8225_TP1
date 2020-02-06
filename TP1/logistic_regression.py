import numpy as np
from sklearn.model_selection import train_test_split

from TP1.side_functions import get_accuracy, softmax, get_grads, get_loss


def logistic_regression(X, y, minibatch_size, lr):
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
    nb_examples = X_train.shape[0]

    for epoch in range(nb_epochs):
        loss = 0
        loss_validation = 0
        grad = 0

        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation, :]
        y_train = y_train[permutation, :]

        for example in range(nb_examples):
            y_pred = softmax(W.dot(X_train[example]))
            grad -= get_grads(y_train[example], y_pred, X_train[example])

            if example != 0 and example % minibatch_size == 0:  # Update W on every minibatch
                grad_moy = grad / minibatch_size
                W -= lr * grad_moy
                grad = 0

        # compute the loss on the train set
        for example in range(nb_examples):
            y_pred = softmax(np.dot(W, X_train[example]))
            loss += get_loss(y_train[example], y_pred)

        loss_avg = loss / nb_examples  # we take the average for the loss on the train set
        losses.append(loss_avg)

        nb_examples = X_validation.shape[0]

        for example in range(0, X_validation.shape[0]):
            y_pred = softmax(np.dot(W, X_validation[example]))
            loss_validation += get_loss(y_validation[example], y_pred)

        losses_validation.append(loss_validation / nb_examples )

        # compute the accuracy on the validation set
        accuracy = get_accuracy(X_validation, y_validation, W)
        accuracies.append(accuracy)

        # select best parameters based ont the validation set
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_W = W

    accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
    return accuracy_on_unseen_data
