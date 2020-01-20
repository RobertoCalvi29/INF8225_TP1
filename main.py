import numpy as np
import matplotlib as plt

# _________________________________________________Question 1___________________________________________________________

prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Pr(Pluie) = {}\n".format(np.squeeze(prob_pluie)))

prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Pr(Arroseur) = {}\n".format(np.squeeze(prob_arroseur)))

watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print("Pr(Watson|Pluie)={}\n".format(np.squeeze(watson)))

holmes = np.zeros(8).reshape(2, 2, 1, 2)
holmes[0, 1, 0, 1] = 0.9
holmes[0, 0, 0, 1] = 0
holmes[1, 0, 0, 1] = 1
holmes[1, 1, 0, 1] = 1
print("Pr(Holmes|Pluie,Arroseur)={}\n".format(np.squeeze(holmes)))

watson[0, :, 1, :]  # prob watson mouille−pluie
watson_mouille = (watson * prob_pluie).sum(0).squeeze()[1]  # prob gazon watson mouille

# a)____________________________________________________________________________________________________________________
# Ce qui revient à
#
# holmes_mouille = holmes[0, 0, 0, 1] * prob_pluie[0, 0, 0, 0] * prob_arroseur[0, 0, 0, 0] + \
#                  holmes[0, 1, 0, 1] * prob_pluie[0, 0, 0, 0] * prob_arroseur[0, 1, 0, 0] + \
#                  holmes[1, 0, 0, 1] * prob_pluie[1, 0, 0, 0] * prob_arroseur[0, 0, 0, 0] + \
#                  holmes[1, 1, 0, 1] * prob_pluie[1, 0, 0, 0] * prob_arroseur[0, 1, 0, 0]

holmes[0, 1, 0, 1]  # prob gazon holmes mouille si arroseur−pluie
holmes_mouille = ((holmes * prob_pluie).sum(0) * prob_arroseur).sum(1).squeeze()[1]
print("Pr(Holmes=1|Pluie,arroseur)={}\n".format(np.squeeze(holmes_mouille)))

# b)____________________________________________________________________________________________________________________
# En réutilisant simplement le résultat obtenue en a) et en se servant des données fournis
# on a Pr(H=1|W=1) = Pr(Holmes=1|Pluie,arroseur) *  Pr(Watson=1|Pluie).

holmes1_watson1 = holmes_mouille * watson_mouille
print("Pr(Holmes=1|Watson=1)={}\n".format(np.squeeze(holmes1_watson1)))

# c)__________________________________________________________________________________________________________
# Toujours en réutilisant en réutilisant simplement le résultat obtenue en a) et en se servant des données fournis
# on a Pr(H=1|W=0) = Pr(Holmes=1|Pluie,arroseur) *  Pr(Watson=0|Pluie).

watson_sec = (watson * prob_pluie).sum(0).squeeze()[0]  # prob gazon watson mouille
holmes1_watson0 = holmes_mouille * watson_sec
print("Pr(Holmes=1|Watson=0)={}\n".format(np.squeeze(holmes1_watson0)))

# d)____________________________________________________________________________________________________________________
# On cherche Pr(H=1|P=0,W=1)
# Pr(H=1|P=0,W=1) = Pr(Holmes=1|P,A) *  Pr(Pluie=0)
#  on a  alors Pr(H=1|P=0,W=1) = Pr(Holmes=1|Watson =1) *  Pr(Pluie=0).

holmes1_P0_watson_1 = holmes1_watson1 * prob_pluie[0]
print("Pr(Holmes=1|P=0,Watson=1)={}\n".format(np.squeeze(holmes1_P0_watson_1)))

# e)____________________________________________________________________________________________________________________
# On cherche Pr(W=1|H=1)
# On applique simplement la règle de Bayes
#  Pr(W=1|H=1) = (Pr(H=1|W=1)*P(W=1))/P(H=1)

watson1_holmes1 = (holmes1_watson1 * watson_mouille) / holmes_mouille
print("Pr(Watson=1|Holmes=1)={}\n".format(np.squeeze(watson1_holmes1)))

# f)____________________________________________________________________________________________________________________
# On cherche Pr(W=1|H=1,A=1)
#  Pr(W=1|H=1,A=1) = Pr(W=1|H=1)*P(A=1) ... pas sur

watson1_holmes1_A1 = watson1_holmes1 * prob_arroseur[1]
print("Pr(Watson=1|Holmes=1)={}\n".format(np.squeeze(watson1_holmes1_A1)))
