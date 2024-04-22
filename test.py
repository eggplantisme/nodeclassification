import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7]
y = [160, 170, 180, 190, 200, 110, 130]

plt.bar(x, y)
plt.title("Height for people")
plt.xlabel("people")
plt.ylabel("height")
plt.show()
