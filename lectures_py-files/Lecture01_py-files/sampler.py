import numpy as np
import matplotlib.pyplot as plt

# Times at which to sample the sine function
t = np.arange(0, 2 * np.pi, 0.1)
A = 1
f0 = 0.5
sig = A * np.sin(2 * np.pi * f0 * t)

# Original signal plot
plt.figure()
plt.plot(t, sig)

# Sampled values
plt.figure()
plt.plot(t, sig, "*")

# Larger sampling period
t = np.arange(0, 2 * np.pi, 0.5)
sig = A * np.sin(2 * np.pi * f0 * t)

plt.figure()
plt.plot(t, sig, "-*")

# Even larger sampling period
t = np.arange(0, 2 * np.pi, 0.8)
sig = A * np.sin(2 * np.pi * f0 * t)

plt.figure()
plt.plot(t, sig, "-*")

plt.show()