import matplotlib.pyplot as plt
import numpy as np

# generate the elevation from (x, y)
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(- x ** 2 - y ** 2)

# generate the grid
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
print('X.shape: ', X.shape)

# fill the color for the contour
# plt.contourf(X, Y, f(X, Y), 8, alpha = 0.75, cmap = plt.cm.hot)

# draw the contour
C = plt.contour(X, Y, f(X, Y), 8, colors = 'green', linewidth = 0.5)
plt.clabel(C, inline = True, fontsize = 10)

# get rid of the axis-lines
plt.xticks(())
plt.yticks(())
plt.show()
