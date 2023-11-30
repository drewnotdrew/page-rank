from data_storage import dataPickle
import numpy as np
import matrix_maker as mm

def main():
    dp = dataPickle()
    v = mm.handle_v_gen(dp)
    #dp.links_to_page #same number of keys as know
    #dp.links_from_page # same number of keys as scanned

if __name__=="__main__":
    main()