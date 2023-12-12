import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class SquaredError():
    def __call__(self, theta, X, y):
        z = np.matmul(X,theta)
        l = np.mean((y - z) ** 2)
        return l

    def gradient(self, theta, X, y):
        z = np.matmul(X,theta)
        g = (-2 * (y - z)).reshape(-1,1) * X
        return np.mean(g, axis = 0)

    def predict(self, theta, X):
        z = np.matmul(X,theta)
        y = (z >= 0.5).astype(int)
        return y

class LogisticLoss():
    def __init__(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x)) 

    def __call__(self, theta, X, y):
        z = np.matmul(X,theta)
        l = -y * np.log(self.sigmoid(z)) - (1-y) * np.log(1-self.sigmoid(z))
        return np.mean(l)

    def gradient(self, theta, X, y):
        z = np.matmul(X,theta)
        g = -(y - self.sigmoid(z)).reshape(-1,1) * X
        return np.mean(g, axis = 0)

    def predict(self, theta, X):
        z = np.matmul(X,theta)
        y = (z >= 0).astype(int)
        return y

class HingeLoss():
    def __call__(self, theta, X, y):
        z = np.matmul(X,theta)
        l = np.maximum(0,1-(2*y-1)*z)
        return np.mean(l)

    def gradient(self, theta, X, y):
        z = np.matmul(X,theta)
        g = (-(2*y-1)).reshape(-1,1) * X
        mask = ((2*y-1) * z >= 1)
        g[mask] = 0
        return np.mean(g, axis=0)

    def predict(self, theta, X):
        z = np.matmul(X,theta)
        y = (z >= 0).astype(int)
        return y
    
X, y = load_iris(return_X_y=True)

X = X[y<2]
y = y[y<2]
n, D = X.shape

X = np.concatenate((X, np.ones((n, 1))), axis=1)
rng = np.random.RandomState(0)
theta = rng.randn(D + 1)

for f in [SquaredError(), LogisticLoss(), HingeLoss()]:
    loss = f(theta, X, y)
    grad = f.gradient(theta, X, y)
    idx = [1,2,3,-1,-2,-3]
    pred = f.predict(theta, X[idx, :])
    print(f"{f.__class__.__name__} : {loss}")
    print(f"Gradient: {grad}")
    print("Predictions for first 3 and last 3 (might be all 1s): ", pred.astype(int) )
    print(f"Gradient shape correct? {np.all(grad.shape == theta.shape)}")
    print(f"Prediction shape correct? {len(pred.shape) == 1 and np.all(pred.shape[0] == len(idx))}\n")

def shuffle_and_batch(X, y, batch_size, rng):
    """Splits both X and y into nearly equal batches"""
    assert X.shape[0] == y.shape[0], 'X and y should have the same number of elements'
    shuffled_idx = rng.permutation(X.shape[0])
    X = X[shuffled_idx, :]
    y = y[shuffled_idx]
    X_batches = np.asarray(np.array_split(X, np.ceil(X.shape[0] / batch_size), axis=0))
    y_batches = np.asarray(np.array_split(y, np.ceil(y.shape[0] / batch_size), axis=0))
    return X_batches, y_batches

def optimize(theta_init, X_raw, y_raw, obj_func, step_size=1,
             max_epoch=100, batch_size=None, rng = None):
    obj_arr = []
    acc_arr = []
    batch_size = batch_size if batch_size is not None else len(X_raw)

    if rng is None:
        rng = np.random.RandomState(0)

    theta = theta_init.copy()
    best_acc = 0
    best_theta = theta
    for i in range(max_epoch):
        X_batches, y_batches = shuffle_and_batch(X_raw, y_raw, batch_size, rng)

        loss_for_each_epoch = 0 
        num_correct = 0

        for batch_idx in range(len(X_batches)):
            gradient = obj_func.gradient(theta, X_batches[batch_idx], y_batches[batch_idx])
            theta = theta - step_size * gradient
            loss_for_each_epoch += obj_func(theta, X_batches[batch_idx], y_batches[batch_idx])
            predictions = obj_func.predict(theta,X_batches[batch_idx])
            num_correct += np.sum(predictions == y_batches[batch_idx])

        avg_loss = loss_for_each_epoch / len(X_batches)
        accuracy = num_correct / len(X_raw)

        if accuracy > best_acc:
            best_acc = accuracy
            best_theta = theta

        obj_arr.append(avg_loss)
        acc_arr.append(accuracy)

        if i == 0 or i == max_epoch - 1:
            print(f'Epoch: {i+1}, Average Loss: {obj_arr[i]}, Accuracy: {acc_arr[i]}')

    return best_theta, obj_arr, acc_arr

obj_func_arr = [SquaredError(), LogisticLoss(), HingeLoss()]
step_sizes = [
    [5e-4, 1e-4],
    [5e-2, 1e-2],
    [1e-2, 5e-2],
]

rng = np.random.RandomState(42)

fig, ax = plt.subplots(3, 2, figsize=(7,7))
plt.subplots_adjust(hspace=0.5)
i = 0

for obj_func, step_size_arr in zip(obj_func_arr, step_sizes): 
    print(f'======= {obj_func.__class__.__name__} =======')
    theta_init = rng.randn(D + 1)

    print(f'-> Running Gradient Descent')
    best_theta, obj_arr, acc_arr = optimize(
        theta_init, X, y, obj_func,
        step_size=step_size_arr[0], max_epoch=100, batch_size=None, rng = rng)
    print(f'\nBest theta: {best_theta}\n')

    print(f'-> Running Mini-Batch Gradient Descent (Batch Size = 10)')
    best_theta_sgd, obj_arr_sgd, acc_arr_sgd = optimize(
        theta_init, X, y, obj_func,
        step_size=step_size_arr[1], max_epoch=100, batch_size=10, rng = rng)
    print(f'\nBest theta_sgd: {best_theta_sgd}')
    print('')

    ax[i][0].set_title(f"Loss plot of {obj_func.__class__.__name__}")
    ax[i][0].set(xlabel="Epoch", ylabel="Loss")
    ax[i][0].plot(np.arange(1, len(obj_arr)+1), obj_arr_sgd, color ="red", label="SGD")
    ax[i][0].plot(np.arange(1, len(obj_arr)+1), obj_arr, color ="blue", label="GD")
    ax[i][0].legend(loc="upper left")
    ax[i][1].set_title(f"Acc plot of {obj_func.__class__.__name__}")
    ax[i][1].set(xlabel="Epoch", ylabel="Accuracy")
    ax[i][1].plot(np.arange(1, len(acc_arr)+1), acc_arr_sgd, color ="red", label="SGD")
    ax[i][1].plot(np.arange(1, len(acc_arr)+1), acc_arr, color ="blue", label="GD")
    ax[i][1].legend(loc="upper left")
    i += 1


plt.show()