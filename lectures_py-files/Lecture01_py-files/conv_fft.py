import numpy as np
import matplotlib.pyplot as plt

# Signal
w = 101
x = np.arange(1, w + 1)

a = np.zeros(w)
center = w // 2
a[center - 5:center + 6] = 1

plt.figure()
plt.plot(x, a)
plt.axis([1, w, -1, 2])

b = a.copy()

# Convolution: retain the central w samples
c = np.convolve(a, b, mode="same")

plt.figure()
plt.plot(x, c)
plt.axis([1, w, -1, 12])

# FFT
A = np.fft.fftshift(np.fft.fft(a))
B = np.fft.fftshift(np.fft.fft(b))

# Comparison
plt.figure()
plt.plot(x, np.abs(A))
plt.axis([1, w, -1, 12])

plt.figure()
plt.plot(x, np.abs(A * B))
plt.plot(x, np.abs(np.fft.fftshift(np.fft.fft(c))), "r--")
plt.axis([1, w, -1, 140])

plt.show()