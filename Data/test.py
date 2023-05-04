import numpy as np
# Finding Duplicate Items in a Python List
#numbers = [1, 2, 3, 2, 5, 3, 3, 5, 6, 3, 4, 5, 7]
numbers=np.loadtxt("time.csv", delimiter=",")
print(numbers[1])
u, c = np.unique(numbers, return_counts=True)
dup = u[c > 1]
print(dup)