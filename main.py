from data_storage import dataPickle
import numpy as np
import generate_eigen

def main():
    dp = dataPickle()
    v = generate_eigen.handle_v_gen(dp)
    #dp.links_to_page #same number of keys as know
    #dp.links_from_page # same number of keys as scanned

if __name__=="__main__":
    main()