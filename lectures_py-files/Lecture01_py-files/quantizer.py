import numpy as np
import matplotlib.pyplot as plt

# Times at which to sample the sine function
t = np.arange(0, 2 * np.pi, 0.1)
A = 3
sig = A * np.sin(t)

# Sampled values
plt.figure()
plt.plot(t, sig, "*")

# 11 boundaries defining 12 intervals
# A = 3 -> [-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
# -> Intervals: (-∞, -3] | (-3, -2.4] | (-2.4, -1.8] | ... | (2.4, 3] | (3, +∞)
partition = A * np.linspace(-1, 1, 11)

# 12 quantization levels, one for each interval
# output level for each interval
codebook = A * np.linspace(-1.2, 1.2, 12)

# Equivalent to MATLAB's quantiz
index = np.digitize(sig, partition, right=True)
quants = codebook[index]

# Compare original and quantized signals
plt.figure()
plt.plot(t, sig, "-", label="Original signal")
plt.plot(t, quants, "r-", label="Quantized signal")
plt.legend()

plt.show()