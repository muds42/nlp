import json
import subprocess
import cPickle
import os
import shutil
import sys

import numpy as np

import utils

def process_and_pickle_wordvecs():
    w2v_file_location = 'w2v/w2v_50d.txt'
    wordvec_dict = {line.split('\t')[0]: np.array(line.split('\t')[1].strip(' \n').split(' '), dtype='float')
                    for line in open(w2v_file_location)}
    utils.write_pickle(wordvec_dict, 'w2v/w2v.pkl')
