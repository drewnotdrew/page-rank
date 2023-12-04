import numpy as np
import numpy.core as npcore

EIGEN_POWER_APPROX_DEPTH = 10

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

def handle_v_gen(db):

    page_list= generate_page_list(db)
    try:
        h = make_weighted_matrix(db,page_list)
        v = eigen_power_approximation(matrix = h, depth = EIGEN_POWER_APPROX_DEPTH)
        print(f"H matrix: \n {h}")

    except npcore._exceptions._ArrayMemoryError as e:
        print("dataset too large to use matrix")
        print(e)
        print("using non matrix math")
        v = np.ones(shape = [1,len(page_list)])/len(page_list)
        v = multiply_by_H(db,page_list,v,depth = EIGEN_POWER_APPROX_DEPTH)

    print(f"Eigen vector \n {v}")
    print((v.tolist()))
    page_rankings = dict(zip(page_list,(v.T.tolist())))
    print(page_rankings)

    
def generate_page_list(db):
    outbound_keys = list(db.links_from_page.keys())
    return outbound_keys

def generate_match_list(pagelist,match_set):
    return [1*(page in match_set) for page in pagelist]

def make_weighted_matrix(db,pagelist):
    size = [len(pagelist),len(pagelist)]
    # print(size)
    h = np.zeros(size,dtype="e")
    for i,key in enumerate(pagelist):
        match_set = db.links_from_page[key]
        row = generate_match_list(pagelist,match_set)
        # print(np.sum(row))
        h[i] = (row/np.sum(row))
        
    #remove the diagonal
    np.fill_diagonal(h,0)
    return h

def get_row_weights(db,page_list):
    return np.array([np.sum(
        generate_match_list(page_list,db.links_from_page[key])
        ) for key in page_list])

def multiply_by_H(db,page_list,v,depth=EIGEN_POWER_APPROX_DEPTH):
    row_weights = get_row_weights(db,page_list)
    for _ in range(depth):
        for i,key in enumerate(page_list):
            match_set = db.links_to_page[key]
            col = generate_match_list(page_list,match_set)
            weighted_col = np.matmul(col,row_weights)
            print(weighted_col)
            v[i] = weighted_col
    return v


if __name__=="__main__":
    import main
    main.main()
