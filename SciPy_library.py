import numpy as np
from scipy import optimize
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt

# Optimization
def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = optimize.minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)

# Signal Processing
t = np.linspace(0, 5, 500, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

# Interpolation
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interpolate.interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 10, num=41, endpoint=True)
ynew = f(xnew)
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

from scipy.optimize import curve_fit

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
ydata = model(xdata, 2.5, 1.3, 0.5)
ydata = ydata + 0.2 * np.random.normal(size=len(xdata))

popt, pcov = curve_fit(model, xdata, ydata)

plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, model(xdata, *popt), '-', label='fit')
plt.legend()
plt.show()

from scipy.fft import fft, fftfreq

# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()


# Generate 1000 random numbers from a normal distribution
data = np.random.normal(0, 1, 1000)

# Generate 1000 random numbers from a uniform distribution
data = np.random.uniform(0, 1, 1000)

# Generate 1000 random integers between 1 and 100 (inclusive)
data = np.random.randint(1, 101, 1000)

from scipy import misc
from scipy import ndimage

# Load an image
face = misc.face(gray=True)

# Blur the image using a Gaussian filter
blurred_face = ndimage.gaussian_filter(face, sigma=3)

# Rotate the image by 45 degrees
rotated_face = ndimage.rotate(face, 45)

# Display the original image, blurred image, and rotated image
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
ax1.imshow(face, cmap='gray')
ax2.imshow(blurred_face, cmap='gray')
ax3.imshow(rotated_face, cmap='gray')
plt.show()

