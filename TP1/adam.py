import numpy as np

# Initialisation
moment_dict = {
    "first_moment": 0,
    "second_moment": 0
}

beta_1 = 0.9
beta_2 = 0.999
epsilon = eps = 1e-8

grad = get_gradient()
m(t) = beta_1 * m[t - 1] + (1 - beta_1) * grad[t]
v(t) = beta_2 * v[t-1] + (1 - beta_2) * grad[t] ** 2
hat_m = m[t] / (1 - beta_1 ** t)
hat_v = v[t]/(1 - beta_2 ** t)

W -= lr * hat_m[:,:,t] / (np.sqrt(hat_v[:,:,t]) + epsilon)


for t in T:
    grad = get_gradient()
    moment_dict = update_moment(moment_dict, gradient)
    corrected_first_moment = compute_bias()
    corrected_second_moment = compute_bias()
    theta = update_theta()
