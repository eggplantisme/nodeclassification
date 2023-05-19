import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# For Detectability With Metadata
rho = np.setdiff1d(np.around(np.arange(0, 1, 0.01), 2), np.array([0, 0.5]))
z = np.setdiff1d(np.around(np.arange(-1, 5, 0.05), 2), np.array([-1, 0]))
X, Y = np.meshgrid(rho, z)
R = 4*np.power(X, 2)*np.power(Y+1, 3)/(np.power(Y, 2)*(Y+1+np.sqrt(1+(np.power(Y, 2)+2*Y)*np.power(1-2*X, 2))))
print(np.inf in R)
# color set
# between_0to1 = ['#FA0202' if i else '' for i in np.logical_and(R < 1, R > 0).tolist()]
# between_neg1to0 = ['#041AFC' if i else '' for i in np.logical_and(R < 0, R > -1).tolist()]
# below_neg1 = ['#B6BFFF' if i else '' for i in (R < -1).tolist()]
# above_1 = ['#FFB1B1' if i else '' for i in (R > 1).tolist()]
C = np.zeros(np.shape(R))
for i in range(np.size(z)):
    for j in range(np.size(rho)):
        if 0 < R[i, j] < 1:
            C[i, j] = 2
        elif -1 < R[i, j] < 0:
            C[i, j] = 1
        elif R[i, j] <= -1:
            C[i, j] = 0
        elif R[i, j] >= 1:
            C[i, j] = 3
        else:
            pass

fig = plt.figure(figsize=(8, 8))
widths = [4]
heights = [4]
spec5 = fig.add_gridspec(ncols=1, nrows=1, width_ratios=widths, height_ratios=heights)
row = 0
col = 0
ax = fig.add_subplot(spec5[row, col], projection='3d')
print(np.shape(X), np.shape(Y), np.shape(R))
print(np.shape(C))
p = ax.scatter3D(X, Y, R, s=0.1, c=C, cmap=cm.coolwarm)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$z$')
ax.set_zlabel(r'$\frac{SNR_2}{SNR_3}$')
cbar = plt.colorbar(p, ticks=[2, 3])
cbar.ax.set_yticklabels(['0~1', '>1'])
plt.show()
