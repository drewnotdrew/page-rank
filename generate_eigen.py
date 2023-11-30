import numpy as np

def eigen_power_approximation (matrix = np.zeros((2,2)), solution = np.zeros(1), depth = 1):
  """
  Recursively calculate the dominant eigenvector of a matrix
  """
  if np.size(solution) == 1:
    num_pages = len(matrix)
    solution_fill = 1/num_pages
    solution = np.full((1, num_pages), solution_fill)

  solution = np.matmul(solution, matrix)
  if depth == 0:
    return solution

  return eigen_power_approximation(matrix = matrix, solution = solution, depth = depth - 1)


solution = eigen_power_approximation(np.matrix('2 -12; 1 -5'), depth = 8)
print(solution)
