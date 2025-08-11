from nns.losses.regression import mse
import numpy as np

yhat = np.array([0.1, 0.4, 0.6])
y = np.array([0.0, 0.5, 1.0])

out = mse(yhat, y, return_grad=True)
print("Loss:", out.value)
print("Grad:", out.grad)
