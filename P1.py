import numpy as np
from matplotlib import pyplot as plt, patches
from sklearn.neighbors import KNeighborsClassifier

Data = np.loadtxt("D2z.txt", dtype=float)
labels = Data[:, 2]
grid = np.arange(-2, 2, 0.1)

print((Data[:, :2]).shape)
print((Data[:, 2]).shape)
nbrs = KNeighborsClassifier(n_neighbors=1).fit(Data[:, :2], Data[:,2])
#print(nbrs.predict(Data[0, :2]))
gridzeros = np.array([-2, -2])
gridones  = np.array([-2, -2])
for x in grid:
    for y in grid:
        if nbrs.predict([[x, y]]) == 0:
            gridzeros = np.vstack((gridzeros, np.array([x, y])))
        else:
            gridones = np.vstack((gridones, np.array([x, y])))
        #gridlabels.append(nbrs.predict([[x, y]]))
gridzeros = gridzeros[1:, :]; gridones = gridones[1:, :]

plt.scatter(gridzeros[:, 0], gridzeros[:, 1], color="blue", s=3)
plt.scatter(gridones[:, 0],  gridones[:, 1], color="red", s=3)
plt.scatter(Data[labels==0, 0], Data[labels==0, 1], color="blue", label="Label 0", marker="o")
plt.scatter(Data[labels==1, 0], Data[labels==1, 1], color="red", label="Label 1", marker="x")
plt.legend()
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("")
plt.savefig("P2.1.png")