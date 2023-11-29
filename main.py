from data_storage import dataPickle

def main():
    dp = dataPickle()
    inbound = dp.links_to_page #same number of keys as know
    outbound = dp.links_from_page # same number of keys as scanned
