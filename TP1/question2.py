import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tqdm import tqdm

from TP1.logistic_regression import logistic_regression

digits = datasets.load_digits()  # Load digits dataset

print(digits.data.shape)

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()


X = digits.data
# X = np.column_stack((X, np.ones(X.shape[0])))
y = digits.target

"""
a) Montrez les résultats pour différents taux d’apprentissage, e.g. 0.1, 0.01, 0.001, 
et différentes tailles de minibatch, e.g. 1, 20, 200, 1000.
"""
nb_epochs = 50
lr_list = [0.1, 0.01, 0.001]
mini_batches_list = [1, 20, 200, 1000]

accuracies_list = []
for lr in lr_list:
    for mini_batch in mini_batches_list:
        accuracy_on_unseen_data = logistic_regression(X, y, mini_batch, lr)
        print("Pour un lr de:", lr, "et une mini-batch de:", mini_batch,
              "; on obtient accuracy=", accuracy_on_unseen_data, "%")
        accuracies_list.append(accuracy_on_unseen_data)


"""
b) Lire l’article de recherche - Adam: a method for stochastic optimization. Kingma, D., & Ba, J. (2015).
 International Conference on Learning Representation (ICLR).4https://arxiv.org/pdf/1412.6980.pdf
Implémentez Adam et présentez des courbes d’apprentissage dans votre rapport où vous comparez l’apprentissage
avec Adam et l’apprentissage avec votre meilleur taux d’apprentissage fixe. Présentez à la fois les courbes 
d’apprentissage sur l’ensemble d’apprentissages et les courbes d’apprentissage pour l’ensemble de validation. 
Comparez les performances finales. N’oubliez pas de soumettre votre code et votre rapport.
"""
