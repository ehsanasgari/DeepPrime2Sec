__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "LLP - DeepPrime2Sec"
__website__ = "https://llp.berkeley.edu/deepprime2sec/"

import argparse
import os
import os.path
import sys
import warnings
from utility.training import training_loop
import yaml

def checkArgs(args):
    '''
        This function checks the input argument and returns the parameters
    '''
    parser = argparse.ArgumentParser()


    # input config #################################################################################################
    parser.add_argument('--config', action='store', dest='config_file', default='sample_configs/model_a.yaml', type=str,
                        help='The config file for secondary structure prediction / please see the examples in the  sample_configs/')


    parsedArgs = parser.parse_args()

    if (not os.access(parsedArgs.config_file, os.F_OK)):
        print("\nError: Permission denied or could not find the config file!")
        return False
    return parsedArgs.config_file

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    res = checkArgs(sys.argv)
    if res != False:
        f = open(res, 'r')
        config=yaml.load(f)
        training_loop(**config)
    else:
        print(res)
        exit()


