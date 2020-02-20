import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from TP1.adam import mini_batch_sgd

digits = datasets.load_digits()  # Load digits dataset

print(digits.data.shape)

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()


X = digits.data
X = np.column_stack((X, np.ones(X.shape[0])))
y = digits.target

"""
a) Montrez les résultats pour différents taux d’apprentissage, e.g. 0.1, 0.01, 0.001, 
et différentes tailles de minibatch, e.g. 1, 20, 200, 1000.
"""

lr_list = [0.1, 0.01, 0.001]
mini_batches_list = [1, 20, 200, 1000]
accuracies_avr_list = []
accuracies_list = []
nb = 1
# for i in range(0, nb):
#     print(i)
#     acc_dict = {}
for lr in lr_list:
    for mini_batch in mini_batches_list:
        losses, loss_validation, accuracies, accuracy_on_unseen_data, best_W = mini_batch_sgd(X,
                                                                                              y,
                                                                                              mini_batch,
                                                                                              lr,
                                                                                              epochs=50)
        print("Learning rate =", lr, " mini-batch size:", mini_batch,
              "; accuracy=", accuracy_on_unseen_data, "%")
#         acc_dict[str(lr + mini_batch)] =accuracy_on_unseen_data
# accuracies_list.append(acc_dict)

# for lr in lr_list:
#     for mini_batch in mini_batches_list:
#         losses, losses_validation, accuracies, naccuracies_list, best_W = [accuracies_list[i][str(lr + mini_batch)]
#                                                                            for i in range(nb)]
#         accuracies_avr = np.mean(naccuracies_list)
#         print("Average for lr = " + str(lr) + " minibatch_size = " + str(mini_batch) + " is :" + str(accuracies_avr))


"""
b) Lire l’article de recherche - Adam: a method for stochastic optimization. Kingma, D., & Ba, J. (2015).
 International Conference on Learning Representation (ICLR).4https://arxiv.org/pdf/1412.6980.pdf
Implémentez Adam et présentez des courbes d’apprentissage dans votre rapport où vous comparez l’apprentissage
avec Adam et l’apprentissage avec votre meilleur taux d’apprentissage fixe. Présentez à la fois les courbes 
d’apprentissage sur l’ensemble d’apprentissages et les courbes d’apprentissage pour l’ensemble de validation. 
Comparez les performances finales. N’oubliez pas de soumettre votre code et votre rapport.
"""

accuracies_avr_list = []
accuracies_list = []

epochs = 100
best_lr = 0.01
best_minibatch_size = 20
nb = 100
a_list = []
aa_list = []
for i in range(10):
    losses, loss_validation, accuracies, accuracy_on_unseen_data, best_W = mini_batch_sgd(X,
                                                                                          y,
                                                                                          best_minibatch_size,
                                                                                          best_lr,
                                                                                          epochs)
    print("Accuracy Without ADAM=", accuracy_on_unseen_data, "%")
    a_list.append(accuracy_on_unseen_data)
    losses_a, loss_validation_a, accuracies_a, accuracy_on_unseen_data_a, best_W_a = mini_batch_sgd(X,
                                                                                                    y,
                                                                                                    best_minibatch_size,
                                                                                                    best_lr,
                                                                                                    epochs,
                                                                                                    adam=True)
    print("Accuracy With ADAM=", accuracy_on_unseen_data_a, "%")
    aa_list.append(accuracy_on_unseen_data_a)
print("Average accuracy on", 10 , " tries without ADAM=", np.mean(a_list), "%")
print("Average accuracy on", 10 , " tries with ADAM=",np.mean(aa_list))


plt.figure()
plt.plot(range(epochs), accuracies, label='Basic regression')
plt.plot(range(epochs), accuracies_a, label='Regression with ADAM')
plt.ylabel('Précision (%)')
plt.xlabel('Epoch')
#plt.title('Précision sur l''ensemble de validation')
plt.legend()
plt.show()

plt.figure()
#plt.plot(range(epochs), losses, linewidth=1, label='Training')
#plt.plot(range(epochs), loss_validation, linewidth=1, linestyle='--', label='Validation')
plt.plot(range(epochs), losses_a, linewidth=1, linestyle=':', label='Training- ADAM')
plt.plot(range(epochs), loss_validation_a, linewidth=1, linestyle='-.', label='Validation - ADAM')
plt.xlabel('Epoch')
plt.ylabel('Average negative log likelihood')
plt.legend()
plt.show()

fig, ax = plt.subplots(2, 5)

ax[0, 0].imshow(best_W[0, 0:64].reshape(8, 8))
ax[0, 1].imshow(best_W[1, 0:64].reshape(8, 8))
ax[0, 2].imshow(best_W[2, 0:64].reshape(8, 8))
ax[0, 3].imshow(best_W[3, 0:64].reshape(8, 8))
ax[0, 4].imshow(best_W[4, 0:64].reshape(8, 8))
ax[1, 0].imshow(best_W[5, 0:64].reshape(8, 8))
ax[1, 1].imshow(best_W[6, 0:64].reshape(8, 8))
ax[1, 2].imshow(best_W[7, 0:64].reshape(8, 8))
ax[1, 3].imshow(best_W[8, 0:64].reshape(8, 8))
ax[1, 4].imshow(best_W[9, 0:64].reshape(8, 8))

plt.show()
