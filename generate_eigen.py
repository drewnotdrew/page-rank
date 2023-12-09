import numpy as np
import numpy.core as npcore
import operator
import matplotlib.pyplot as plt

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
        page_list.remove('/wiki/Portal:Current_events')
    except:
        None
    try:
        page_list.remove('/wiki/Main_Page')
    except:
        None
    
    try:
        page_list.remove('/wiki/Wayback_Machine')
    except:
        None
    '''
    try:
        page_list.remove('/wiki/Geographic_coordinate_system')
    except:
        None#'''

    try:
        h = make_weighted_matrix(db,page_list)
        #raise npcore._exceptions._ArrayMemoryError(h.shape, h.dtype)
        v = eigen_power_approximation(matrix = h, depth = EIGEN_POWER_APPROX_DEPTH)
        print(f"H matrix: \n {h}")

    except npcore._exceptions._ArrayMemoryError as e:
        print("dataset too large to use matrix")
        print(e)
        print("using non matrix math")
        num_pages = len(page_list)
        solution_fill = 1/num_pages
        v = np.full((1, num_pages), solution_fill)
        #v = np.ones(shape = [1,len(page_list)])/len(page_list)
        v = multiply_by_H(db,page_list,v,depth = EIGEN_POWER_APPROX_DEPTH)

    print(f"Eigen vector \n {v}")
    print((v.tolist()))
    page_rankings = dict(zip(page_list,(v.T.tolist())))
    print(page_rankings)
    print(sorted(page_rankings.items(),key=operator.itemgetter(1)))
    show_rankings(page_rankings)
    ranked = list(sorted(page_rankings.items(),key=operator.itemgetter(1)))
    ranked.reverse()
    for i in range(10):
        print(ranked[i][0])



def show_rankings(page_rankings,shown = 10):
    plt.figure(figsize=[11,8.5])
    data = dict(sorted(page_rankings.items(),key=operator.itemgetter(1))[-shown:])
    names = [name[6:] for name in list(data.keys())]
    values = np.array(list(data.values())).flatten()
    bars = plt.barh(range(len(data)), values, tick_label=names)
    i = 0
    color_list = ['#4385f5']#,'#dc4438','#109d59','#f5b401']
    for bar in bars:
        bar.set_color(color_list[i])
        i=(i+1)%len(color_list)
    #plt.yticks(rotation='vertical')
    plt.text(-2,0,"")
    plt.xlabel("Page Rank Value")
    plt.show()
    None

def generate_page_list(db):
    outbound_keys = list(db.links_from_page.keys())
    return outbound_keys

def generate_match_list(pagelist,match_set):
    return [(page in match_set) for page in pagelist]

def generate_match_list_v2(pagelist,match_set):
    # not functional atm
    out = np.zeros(shape=[len(pagelist),1])
    for page in match_set:
        return [(page in match_set) for page in pagelist]

def make_weighted_matrix(db,pagelist):
    size = [len(pagelist),len(pagelist)]
    # print(size)
    #row_weights = get_row_weights(db,pagelist)
    h = np.zeros(size)
    for i,key in enumerate(pagelist):
        match_set = db.links_from_page[key]
        row = generate_match_list(pagelist,match_set)
        s = np.sum(row)
        #w = row_weights[i]
        #t = np.multiply(row,s)
        if s<1:
            None
            h[i] = 0
            continue
        h[i] = (row/s)
        
    #remove the diagonal
    np.fill_diagonal(h,0)
    return h

def get_row_weights(db,page_list):
    out = np.zeros(shape=(len(page_list)))
    for i,key in enumerate(page_list):
        s = np.sum(generate_match_list(page_list,db.links_from_page[key]))
        if s <=0 :
            out[i] = 0
            continue
        out[i] = 1/s
    return out

def multiply_by_H(db,page_list,v,depth=EIGEN_POWER_APPROX_DEPTH):
    row_weights = get_row_weights(db,page_list)
    out = np.zeros(shape = v.shape)
    #h = make_weighted_matrix(db,page_list)
    for _ in range(depth):
        for i,key in enumerate(page_list):
            match_set = db.links_to_page[key]
            col = generate_match_list(page_list,match_set)
            col[i] = False
            weighted_col = np.multiply(col,row_weights)
            # test_col = h[:, i]

            # for j, (t,y) in enumerate(zip(weighted_col,test_col)):
            #     if not abs(t - y)<.001:
            #         print("error with row:",j,"column:",i)
            #b = weighted_col.tolist()
            #c = v.tolist()
            #a = np.multiply(v,weighted_col).tolist()
            #s = np.sum(a)
            out[0][i] = np.sum(np.multiply(v,weighted_col.T))
        v = out
    return v


if __name__=="__main__":
    import main
    main.main()
