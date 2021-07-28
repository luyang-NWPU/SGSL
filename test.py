import numpy as np 



matrix_a = np.random.randint(10, size=(2,3))
matrix_b = np.random.randint(10, size=(2,3))
matrix_c = np.random.randint(10, size=(3,4))

print('matrix_a', matrix_a)
print('matrix_b', matrix_b)
print('matrix_c', matrix_c)

out1 = np.matmul(matrix_a, matrix_c) + np.matmul(matrix_b, matrix_c)
out2 = np.matmul(matrix_a+matrix_b, matrix_c)
print(out1)
print(out2)
