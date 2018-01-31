import pdb
import logging
import cPickle

import tensorflow as tf
import utils as cr_utils
import logistic_regression_baseline as lrb
import nn_baseline as mnn

import logging
import numpy as np

from q3_util import Progbar, minibatches

import argparse
import sys
import time
import logging

ALL_FIELDS = [u'same-speaker', 'm2 head word', 'm2 first word', 'bias',
    'mention distance', u'relaxed-string-match', 'sentence distance = 1',
    'sentence distance = 2', 'sentence distance = 3',
    'sentence distance = 4', 'sentence distance = 5',
    'sentence distance > 5',
    'heads', 'm1 first word', 'sentence distance', 'm2 last word',
    'm1 last word', u'exact-string-match', 'm2 prev word',
    u'mention-is-antecedent-speaker', 'm2 next word',
    'mention distance > 5', 'm1 prev word',
    'm1 head word', 'm1 next word', u'relaxed-head-match',
    u'antecedent-is-mention-speaker']
CUSTOM_FIELDS = ['head_dot', 'first_dot', 'last_dot', 'three_dot']
TEXT_FIELDS = ['m2 head word', 'm2 first word', 'heads', 'm1 first word',
    'm2 last word', 'm1 last word', 'm2 prev word', 'm2 next word',
    'm1 prev word', 'm1 head word', 'm1 next word']

EMBED_LENGTH = 50
CUSTOM_FEAT_LENGTH = len(ALL_FIELDS) + len(CUSTOM_FIELDS) + \
    (EMBED_LENGTH - 1) * len(TEXT_FIELDS)  # text fields already counted once

# dict of 0 everywhere and 1 where key is true
def create_name_index_dict():
    return_val = {}
    arr_len = CUSTOM_FEAT_LENGTH
    dft = np.zeroes(arr_len)
    counter = 0
    for field in ALL_FIELDS:
        temp_len = EMBED_LENGTH if field in TEXT_FIELDS else 0
        tmp_arr = dft.copy()
        tmp_arr[counter:counter+temp_len] = 1
        return_val[field] = tmp_arr
        counter += temp_len
    for field in CUSTOM_FIELDS:
        tmp_arr = dft.copy()
        tmp_arr[counter] = 1
        return_val[field] = tmp_arr
        counter += 1

    return return_val

# easy to do a find-not-match type of setup
def not_indexer(not_arg):
    assert not_arg in ('m1', 'm2')
    # embed_length = EMBED_LENGTH
    output_array = np.array([], dtype=np.int64)
    for field in ALL_FIELDS:
        num = EMBED_LENGTH if field in TEXT_FIELDS else 1
        val = 1 if -field.find(not_arg) else 0  # this means NOT ARG = 1
        output_array = np.append(output_array, [val]*num)
    for field in CUSTOM_FIELDS:  # include all custom in both
        output_array = np.append(output_array, [1]*1)
    return output_array


def convert_train_examples(train_inp, wordvec_dict):
    def wordvec_func(wds):
        words = wds.split(' ')
        return mnn.multi_word_function([wordvec_dict.get(x, np.zeros(50)) for x in words])
    ret_val = []
    prog = Progbar(len(train_inp))
    counter = 0
    for inp in train_inp:
        tmp = np.array([], dtype=np.float16)
        for field in ALL_FIELDS:  # i am assuming this will enforce ordering
            if field not in inp.keys():
                tmp = np.append(tmp, np.array([0])) # this is to handle other sentence dist features
            elif field not in TEXT_FIELDS:
                tmp = np.append(tmp, np.array(inp[field]))
            else:
                words = inp[field].split(' ')
                tmp = np.append(tmp, mnn.multi_word_function(
                    [wordvec_dict.get(x, np.zeros(50)) for x in words]))
        for field in CUSTOM_FIELDS:
            # ['head_dot', 'first_dot', 'last_dot', 'three_dot']
            # import pdb; pdb.set_trace()
            if field == 'head_dot':
                tmp = np.append(tmp,
                    np.dot(wordvec_func(inp['m1 head word']), wordvec_func(inp['m2 head word'])))
            elif field == 'first_dot':
                tmp = np.append(tmp,
                    np.dot(wordvec_func(inp['m1 first word']), wordvec_func(inp['m2 first word'])))
            elif field == 'last_dot':
                tmp = np.append(tmp,
                    np.dot(wordvec_func(inp['m1 last word']), wordvec_func(inp['m2 last word'])))
            elif field == 'three_dot':
                tmp = np.append(tmp,
                    np.dot(wordvec_func(inp['m1 first word']), wordvec_func(inp['m2 first word'])) +
                    np.dot(wordvec_func(inp['m1 head word']), wordvec_func(inp['m2 head word'])) +
                    np.dot(wordvec_func(inp['m1 last word']), wordvec_func(inp['m2 last word']))
                )
        ret_val.append(tmp)
        counter += 1
        prog.update(counter)
    return ret_val

def prepare_custom_data():
    train_x, train_y = lrb.build_examples("train")
    dev_x, dev_y = lrb.build_examples("dev")
    wordvec_dict = cr_utils.load_pickle('w2v/w2v.pkl')
    mod_train_x = convert_train_examples(train_x, wordvec_dict)
    mod_dev_x = convert_train_examples(dev_x, wordvec_dict)

    cr_utils.write_pickle((mod_train_x, np.array(train_y), mod_dev_x,
        np.array(dev_y)), "./w2v/nn_nominal_full_dataset_custom.pkl")
    # create mini-training and eval sets for rapid iterations
    cr_utils.write_pickle((mod_train_x[:50000], np.array(train_y[:50000]),
        mod_dev_x[:10000], np.array(dev_y[:10000])), "./w2v/nn_nominal_mini_dataset_custom.pkl")



def prepare_test_data():
    test_x, test_y = lrb.build_examples("test", downsample_prob=1)
    wordvec_dict = cr_utils.load_pickle('w2v/w2v.pkl')
    mod_test_x = convert_train_examples(test_x, wordvec_dict)
    cr_utils.write_pickle((mod_test_x, np.array(test_y)), "./w2v/nn_nominal_TEST_NDS_custom.pkl")


def prepare_100_custom_data():
    # crashes due to memory overload
    train_x, train_y = lrb.build_examples("train", downsample_prob=1.1)
    dev_x, dev_y = lrb.build_examples("dev", downsample_prob=1.1)
    wordvec_dict = cr_utils.load_pickle('w2v/w2v.pkl')
    mod_train_x = convert_train_examples(train_x, wordvec_dict)
    mod_dev_x = convert_train_examples(dev_x, wordvec_dict)

    cr_utils.write_pickle((mod_train_x, np.array(train_y), mod_dev_x,
        np.array(dev_y)), "./w2v/nn_nominal_full_100_dataset_custom.pkl")
    # create mini-training and eval sets for rapid iterations
    cr_utils.write_pickle((mod_train_x[:50000], np.array(train_y[:50000]),
        mod_dev_x[:10000], np.array(dev_y[:10000])), "./w2v/nn_nominal_mini_100_dataset_custom.pkl")


if __name__ == "__main__":
    # prepare_custom_data()
    prepare_test_data()
