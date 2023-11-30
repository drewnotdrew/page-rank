import numpy as np
import numpy.core as npcore

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

def handle_v_gen(db):

    page_list= generate_page_list(db)
    try:
        h = make_weighted_matrix(db,page_list)
        print(h)
    except npcore._exceptions._ArrayMemoryError as e:
        print("dataset too large to use matrix")
        print(e)

    
def generate_page_list(db):
    outbound_keys = list(db.links_from_page.keys())
    return outbound_keys

def generate_match_list(pagelist,match_set):
    return [1*(page in match_set) for page in pagelist]


def make_weighted_matrix(db,pagelist):
    size = [len(pagelist),len(pagelist)]
    print(size)
    h = np.zeros(size,dtype="e")
    for i,key in enumerate(pagelist):
        match_set = db.links_from_page[key]
        row = generate_match_list(pagelist,match_set)
        print(np.sum(row))
        h[i] = (row/np.sum(row))
        
    #remove the diagonal
    np.fill_diagonal(h,0)
    return h


if __name__=="__main__":
    import main
    main.main()
