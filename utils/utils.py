import numpy as np

class NanError(Exception):
    pass

# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        #print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
        print("{}:\t{}".format(k, str(v)))
    print("########################################################################")

def print_dict(sample, prefix=''):
    if (prefix==''):
        print("*"*20)
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print("{} {}: {}".format(prefix, k, v.shape))
        elif isinstance(v, dict):
            print_dict(v, prefix=k) # recursive print
    if (prefix==''):
        print("*"*20)