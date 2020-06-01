import torch
import cv2 as cv
import numpy as np

np.random.seed(42)
a = np.random.rand(1)
b = np.random.rand(1)
x = np.random.rand(100,1)
# ground truth a = 1, b = 2 
y = 1 + 2 * x + 0.1 * np.random.rand(100,1)
idx = np.arange(100) #1-100

np.random.shuffle(idx)
train_idx = idx[:80]
val_idx = idx[80:]
# generate train and validation test
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


lr = 1e-1
n_epochs = 1000
for epoch in range(n_epochs):
    y_predict = a + b * x_train
    error = (y_train - y_predict)
    # MSE loss
    loss = (error ** 2).mean() 
    # compute gradients 
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    # Update a,b
    a = a - lr * a_grad
    b = b - lr * b_grad
print(a)
print(b)
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train,y_train)
print(linr.intercept_, linr.coef_[0])

