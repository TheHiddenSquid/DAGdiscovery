import numpy as np
import ges
from scipy import stats
import matplotlib.pyplot as plt

samples = 1000
N1 = stats.norm(0,1).rvs(size=samples)
N2 = stats.norm(0,0.5).rvs(size=samples)
N3 = stats.norm(0,0.1).rvs(size=samples)
lambda12 = 10
lambda23 = -5


node1 = N1
node2 = [lambda12*N1[i] + N2[i] for i in range(samples)]
node3 = [lambda23*N2[i] + N3[i] for i in range(samples)]

data = np.transpose(np.array([node1, node2, node3]))

plt.hist(node1, bins=20, density=True)
plt.hist(node2, bins=20, density=True)
plt.hist(node3, bins=20, density=True)
plt.show()

A0 = np.array([[0,1,0],[0,0,1],[0,0,0]])
res = ges.fit_bic(data=data, A0=A0)
print(res)