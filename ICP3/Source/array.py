import numpy as np

x = np.random.uniform(1,20,20) # To print array with float number iin between 1 and 20
print(x)
print("----------------------------------------------")
x = np.reshape(x, (4, 5)) # Reshaping the array to 4X5
print(x)
print("----------------------------------------------")
#Finding the maximum of each row and replacing it with zero
z = np.where(x == np.max(x, axis=1).reshape(-1,1), 0, x)
print(z)