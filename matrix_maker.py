import numpy as np
import numpy.core as npcore

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
        #print(np.sum(row))
        h[i] = (row/np.sum(row))
        
    #remove the diagonal
    np.fill_diagonal(h,0)
    return h

def get_row_weights(db,page_list):
    return np.array([np.sum(
        generate_match_list(page_list,db.links_from_page[key])
        ) for key in page_list])

def multiply_by_H(db,page_list,v):
    row_weights = get_row_weights(db,page_list).T
    for i,key in enumerate(page_list):
        match_set = db.links_to_page[key]
        col = generate_match_list(page_list,match_set)
        weighted_col = np.multiply(col,row_weights)
        print(weighted_col)
        v[i] = np.sum(np.multiply(weighted_col,row_weights.T))


if __name__=="__main__":
    import main
    main.main()