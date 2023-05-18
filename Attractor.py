import matplotlib.pyplot as plt
from mpmath import fp
from datetime import datetime
import numpy as np
import mpmath as mp

mp.dps = 100

sol = []
N = 15
a_start = mp.mpc(0.5)
# N = int(input("Enter depth: "))
# a_start = float(input("Enter starting a: "))

now = datetime.now()
date_time = now.strftime("%m%d%Y-%H%M%S")


class Node:
    def __init__(self, data, address):
        self.data = data
        self.leftChild = None
        self.rightChild = None
        self.address = address
        self.steps = 0

    def insert(self, sol1, sol2, parent_address):
        self.steps += 1
        if self.address == parent_address:
            self.leftChild = Node(sol1, parent_address + [1])
            self.rightChild = Node(sol2, parent_address + [2])
            # stop walking up and down the tree
            return True
        else:
            if self.leftChild and self.rightChild:
                return True if self.leftChild.insert(sol1, sol2, parent_address) else self.rightChild.insert(sol1, sol2, parent_address)


    def walk(self):
        if not (self.leftChild and self.rightChild):
            sol.append((self.data, self.address))
        else:
            sol.append((self.data, self.address))
            self.leftChild.walk()
            self.rightChild.walk()


a_lst = []
z_lst = []


def f(z, a):
    return z * (z - a) / (1 - a * z)


def build_list_a():
    a = a_start
    a_lst.append(a)
    for i in range(N):
        a = mp.mpc(a*4/7 * a ** 2)
        a_lst.append(a)


def solve(f_z, n, path):
    # z2 + (f(z) - 1)az - f(z) = 0
    if n < 1:
        return
    a = 1
    b = mp.mpc((f_z - 1) * a_lst[n - 1])
    c = mp.mpc(-f_z)
    d = mp.mpc(((b ** 2) - (4 * a * c)))

    sol1 = mp.mpc(-b - mp.sqrt(d) / (2 * a))
    sol2 = mp.mpc(-b + mp.sqrt(d) / (2 * a))

    z_lst.append(sol1)
    z_lst.append(sol2)

    sol_tree.steps = 0
    sol_tree.insert(sol1, sol2, path)
    # print(sol_tree.steps)

    solve(sol1, n - 1, path + [1])
    solve(sol2, n - 1, path + [2])


build_list_a()
starting_point = np.nextafter(0,1)
# starting_point = 0
sol_tree = Node(starting_point, [0])
solve(starting_point, N, [0])  # n-1 because a_lst size = N+1
sol_tree.walk()
sol.sort(key=lambda x: x[1].__len__())
print("Total solutions: ", z_lst.__len__())

plt.scatter([x[0].real for x in sol], [x[0].imag for x in sol],
            c=[(x[1].__len__()-1) * 100 / N for x in sol], cmap='seismic', s=2, alpha=1)
plt.xlabel('')
plt.ylabel('')
plt.colorbar()

circle2 = plt.Circle((0, 0), 1, linewidth=0.5, color='black', fill=False)
ax = plt.gca()
ax.add_patch(circle2)
plt.axline((0, 0), (0, 1), linewidth=0.5, color='black')
plt.axline((0, 0), (1, 0), linewidth=0.5, color='black')


resolution_value = 1200
plt.savefig(f"d_{N}_a_{a_start}_{date_time}.png", format="png", dpi=resolution_value)
plt.show()
