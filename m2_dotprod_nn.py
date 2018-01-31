import nn_baseline as m
import tensorflow as tf
import numpy as np
import utils as cr_utils
import logistic_regression_baseline as lrb
import custom_feature_append as custom_feat
import sklearn as sk
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from q3_util import Progbar, minibatches

import argparse
import sys
import time
import logging
import random

logger = logging.getLogger("dot_prod_nn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# find the threshold that maximizes f1 on training data
def f1_tuner(train_labels, train_predictions, max_val=1):
    print "running f1 tuner"
    threshold = 0
    retval = 0.5
    check_range = (np.logspace(0, max_val, num=30, base=np.e)-1)/(np.e-1)
    for val in check_range:
        tmp = lrb.f1_score(train_labels, np.hstack(train_predictions)>val)
        # print "f1 value is: %.3f, when the pct cutoff is %.4f" % (tmp, val)
        if tmp > threshold:
            threshold = tmp
            retval = val
    print "going to use %.4f as the threshold" % retval
    final_trn_f1 = lrb.f1_score(train_labels, np.hstack(train_predictions)>retval)
    print "the best train f1 is %.2f" % final_trn_f1
    return retval

def calibration(labels, predictions, assumed_downsample=0.02):
    npos = sum(labels)
    nneg = sum(labels==0)
    inferred_true_rate = npos / (npos + nneg * 1/assumed_downsample)
    pctile = (1 - inferred_true_rate) * 100  # in pcts for np
    return np.percentile(predictions, pctile)

def print_errs(labels, predictions, aligned_mention_info):
    def get_2_mentions(ami, idx):
        return ami[idx]["m1 full mention"] + "; " + ami[idx]["m2 full mention"]
    top_10_false_neg = np.lexsort((-predictions, labels))[-10:]
    print "Errors made (false negatives):"
    for idx in top_10_false_neg:
        print "Label: %d -- Prb: %.2f -- mention: %s" % \
            (labels[idx], predictions[idx], get_2_mentions(aligned_mention_info, idx))
    top_10_false_pos = np.lexsort((predictions, -labels))[-10:]
    print "Errors made (false positives):"
    for idx in top_10_false_pos:
        print "Label: %d -- Prb: %.2f -- mention: %s" % \
            (labels[idx], predictions[idx], get_2_mentions(aligned_mention_info, idx))
    top_10_preds = np.lexsort((labels,predictions))[-10:]
    print "Top 10 predictions:"
    for idx in top_10_preds:
        print "Label: %d -- Prb: %.2f -- mention: %s" % \
            (labels[idx], predictions[idx], get_2_mentions(aligned_mention_info, idx))


# easy to do a find-not-match type of setup
def not_indexer(not_arg):
    assert not_arg in ('m1', 'm2')
    embed_length = 50
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
    TEXT_FIELDS = ['m2 head word', 'm2 first word', 'heads', 'm1 first word',
        'm2 last word', 'm1 last word', 'm2 prev word', 'm2 next word',
        'm1 prev word', 'm1 head word', 'm1 next word']
    output_array = np.array([], dtype=np.int64)
    for field in ALL_FIELDS:
        num = 50 if field in TEXT_FIELDS else 1
        val = 1 if -field.find(not_arg) else 0  # this means NOT ARG = 1
        output_array = np.append(output_array, [val]*num)

    output_array = np.append(output_array, [0]*4)  # 0-mask custom features
    return output_array


class DotProdConfig:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    custom_features = True
    # n_feat = 11*50 + 16 if not custom_features else custom_feat.CUSTOM_FEAT_LENGTH  #add 4 custom features
    n_feat = custom_feat.CUSTOM_FEAT_LENGTH  #add 4 custom features
    int_vec_size = 100  # how much gets dotprod at the last set
    hidden_size, hidden_size2 = 500, 200
    # hidden_size2 = 200
    # hidden_size3 = 100
    n_epochs = 15
    # lr = 0.001
    lr = 0.0015
    batch_size = 300
    model_one = False  # default is assumed to be model 2.. messy, messy
    model_three = True
    model_three_prb = True
    cross_entropy_weighted_poswt = 10
    # regularization_l2_weight = 0.0001
    # dropout_rate = 0.50
    verbose_error_detail = False  # wont work on GPU, for now

    def __repr__(self):
        pub_attr = [f for f in dir(self) if not f.startswith('_')]
        return {attr: getattr(self, attr) for attr in pub_attr}.__repr__()


class DotProdNN(m.VerySimpleNN):
    def sub_pred_op(self, x, name):
        cf = self.config
        W = tf.get_variable("W_%s" % name, shape=(cf.n_feat, cf.hidden_size),
            initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2_%s" % name, shape=(cf.hidden_size, cf.hidden_size2),
            initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable("b1_%s" % name, initializer=tf.zeros(cf.hidden_size, ))
        b2 = tf.get_variable("b2_%s" % name, initializer=tf.zeros(cf.hidden_size2, ))
        h = tf.nn.relu(tf.matmul(x, W) + b1)

        if hasattr(cf, 'dropout_rate'):
            hdrop = tf.nn.dropout(h, keep_prob = cf.dropout_rate)
            h2 = tf.nn.relu(tf.matmul(hdrop, W2) + b2)
        else:
            h2 = tf.nn.relu(tf.matmul(h, W2) + b2)
        # h2 = tf.nn.relu(tf.matmul(h, W2) + b2)

        if hasattr(cf, 'hidden_size3'):
            W3 = tf.get_variable("W3_%s" % name, shape=(cf.hidden_size2, cf.hidden_size3),
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3_%s" % name, initializer=tf.zeros(cf.hidden_size3, ))
            ushape = cf.hidden_size3
            h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
        else:
            ushape = cf.hidden_size2

        U = tf.get_variable("U_%s" % name, shape=(ushape, cf.int_vec_size),
            initializer=tf.contrib.layers.xavier_initializer())

        if hasattr(cf, 'hidden_size3'):
            pred_vec = tf.matmul(h3, U)
        else:
            pred_vec = tf.matmul(h2, U)

        return pred_vec

    def add_prediction_op(self):
        x = self.inputs_placeholder
        cf = self.config
        if hasattr(cf, 'model_one') and cf.model_one == True:
            pred_vec = self.sub_pred_op(x, '')
            U = tf.get_variable("U", shape=(cf.int_vec_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            pred_full = tf.reshape(tf.matmul(pred_vec, U), [-1])
            return tf.transpose(tf.sigmoid(pred_full))

        if hasattr(cf, 'custom_features') and cf.custom_features == True:
            ni_func = custom_feat.not_indexer
        else:
            ni_func = not_indexer
        x_m1 = x * ni_func('m2')  # inefficient, but send 0s through
        x_m2 = x * ni_func('m1')

        pred_m1 = self.sub_pred_op(x_m1, 'm1')
        pred_m2 = self.sub_pred_op(x_m2, 'm2')

        top_bias1 = tf.get_variable("top_bias1", shape=(cf.int_vec_size, ),
            initializer=tf.contrib.layers.xavier_initializer())
        top_bias2 = tf.get_variable("top_bias2", shape=(cf.int_vec_size, ),
            initializer=tf.contrib.layers.xavier_initializer())

        # dot_prod = tf.reduce_sum(tf.multiply(pred_m1 + top_bias1, pred_m2 + top_bias2),1)
        dot_prod = tf.reduce_mean(tf.multiply(pred_m1 + top_bias1, pred_m2 + top_bias2),1)

        if hasattr(cf, 'model_three') and cf.model_three == True:
            pred_full_vec = self.sub_pred_op(x, '')
            U = tf.get_variable("U", shape=(cf.int_vec_size, 1),
                initializer=tf.contrib.layers.xavier_initializer())
            pred_full = tf.reshape(tf.matmul(pred_full_vec, U), [-1])

            if hasattr(cf, 'model_three_prb') and cf.model_three_prb == True:
                xHP = tf.get_variable("xHP", shape=(cf.n_feat,1),
                    initializer=tf.contrib.layers.xavier_initializer())
                prb = tf.reshape(tf.sigmoid(tf.matmul(x, xHP)), [-1])
                pred = prb * tf.sigmoid(dot_prod) + (1 - prb) * tf.sigmoid(pred_full)
            else:
                pred = tf.sigmoid(dot_prod + pred_full)
        else:  # model 2
            pred = tf.sigmoid(dot_prod)  # move sigmoid to loss op, b/c of CE impl

        return tf.transpose(pred) #state # preds

    def compute_regularization_func(self, type='L2'):
        cf = self.config
        loss = 0
        var_base_names = ['W_', 'W2_', 'b1_', 'b2_']
        if hasattr(cf, 'hidden_size3'):
            var_base_names += ['W3_', 'b3_']

        var_end_names = ['m1', 'm2']
        if hasattr(cf, 'model_three'):
            var_end_names += ['']

        list_of_vars = [a+b for a in var_base_names for b in var_end_names]

        tf.get_variable_scope().reuse_variables()
        if type == 'L2':
            # generic form of beta * tf.nn.l2_loss(weights); weights = variable
            for var in list_of_vars:
                loss += tf.nn.l2_loss(tf.get_variable(var))

        return loss

    def add_loss_op(self, preds):
        y = self.labels_placeholder
        # loss = tf.nn.l2_loss(preds-y)
        ce_wt = self.config.cross_entropy_weighted_poswt
        logits = tf.log(tf.maximum(tf.div(preds, tf.maximum(1-preds, 0.0001)), 0.00001))  # need to prevent 0 or inf
        loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(logits, y, ce_wt if ce_wt is not None else 1))

        if hasattr(self.config, 'regularization_l2_weight'):
            loss += self.config.regularization_l2_weight * self.compute_regularization_func(type='L2')
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(self, sess, train):
        #prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            # import pdb; pdb.set_trace()
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
            #prog.update(i + 1, [("train loss", loss)])

        return losses

    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        # if inputs is None:
        #     inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        labels = []
        # prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            # batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            # prog.update(i + 1, [])
        return preds


def dot_prod_nn(config=None):
    random.seed(0)
    if config is None:
        config = DotProdConfig()
    report = None
    datafile = "./w2v/nn_nominal_full_dataset_custom.pkl"

    train_x, train_y, dev_x, dev_y = m.cr_utils.load_pickle(datafile)
    # this err eval has custom feats, len 570
    if hasattr(config, "verbose_error_detail") and config.verbose_error_detail == True:
        dev_x, dev_y, dev_z = m.cr_utils.load_pickle("./w2v/nn_nominal_downsample_WERR.pkl")

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = DotProdNN(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            # model.fit(session, [train_x, train_y])
            model.fit(session, zip(train_x, train_y))
            output = model.output(session, zip(dev_x))
            # score the predictions..
            f1_cutoff = f1_tuner(train_y, model.output(session, zip(train_x)))
            # y_pred = np.floor(np.hstack(output)*2)  # 50% cutoff; maybe can be calibrated
            y_pred = np.hstack(output)>f1_cutoff
            # logger.info("\nSSE on predictions is: %f" % (np.sum(np.square(y_pred-dev_y))))
            auc_roc = roc_auc_score(dev_y, np.hstack(output))
            print "LOG_AR_MS: F1 score: {:.2f}".format(lrb.f1_score(dev_y, y_pred))
            print "LOG_AR_MS: F1 score: {:.2f} for 50pct cutoff".format(lrb.f1_score(dev_y, np.hstack(output)>0.5))
            print "LOG_AR_MS: AUC score: {:.2f}".format(lrb.average_precision_score(dev_y, np.hstack(output)))
            print "LOG_AR_MS: AUC(ROC) score: {:.2f}".format(auc_roc)
            print "LOG_AR_MS: Precision", sk.metrics.precision_score(dev_y, y_pred)
            print "LOG_AR_MS: Recall", sk.metrics.recall_score(dev_y, y_pred)
            print sk.metrics.confusion_matrix(dev_y, y_pred)
            if hasattr(config, "verbose_error_detail") and config.verbose_error_detail == True:
                print_errs(dev_y, np.hstack(output), dev_z)
                cr_utils.write_pickle((dev_y, np.hstack(output), dev_z), "./w2v/nn_error_align_cf_%s.pkl" % config.custom_features)


def dot_prod_nn_fulldataset(config=None):
    random.seed(0)
    if config is None:
        config = DotProdConfig()
    report = None

    datafile = "./w2v/nn_nominal_train_FULL.pkl"  # actually just 900 docs; no downsample
    train_x, train_y = m.cr_utils.load_pickle(datafile)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = DotProdNN(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, zip(train_x, train_y))
            f1_cutoff = f1_tuner(train_y, model.output(session, zip(train_x)), max_val=0.18)
            del train_x, train_y  # free up memory
            dev_x, dev_y = m.cr_utils.load_pickle("./w2v/nn_nominal_dev_FULL.pkl")
            output = model.output(session, zip(dev_x))
            # score the predictions..
            y_pred = np.hstack(output)>f1_cutoff
            print "F1 score: {:.2f}".format(lrb.f1_score(dev_y, y_pred))
            print "F1 score: {:.2f} for 50pct cutoff".format(lrb.f1_score(dev_y, np.hstack(output)>0.5))
            print "AUC score: {:.2f}".format(lrb.average_precision_score(dev_y, np.hstack(output)))


def dot_prod_nn_on_TEST(config=None, eval_dataset="DOWNSAMPLE"):
    random.seed(0)
    if config is None:
        config = DotProdConfig()
    report = None
    datafile = "./w2v/nn_nominal_full_dataset_custom.pkl"
    train_x, train_y, dev_x, dev_y = m.cr_utils.load_pickle(datafile)
    if eval_dataset == "DOWNSAMPLE":
        test_x, test_y = m.cr_utils.load_pickle("./w2v/nn_nominal_TEST_custom.pkl")
    elif eval_dataset == "ALL":
        test_x, test_y = m.cr_utils.load_pickle("./w2v/nn_nominal_TEST_NDS_custom.pkl")
    else:
        raise("need either DOWNSAMPLE of ALL for eval_dataset")

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = DotProdNN(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, zip(train_x, train_y))
            output = model.output(session, zip(test_x))
            # score the predictions..
            if eval_dataset == "DOWNSAMPLE":
                f1_cutoff = f1_tuner(train_y, model.output(session, zip(train_x)))
            else:  # calibration
                f1_cutoff = calibration(train_y, np.hstack(output))
                print "post-calibration cutoff is %f" % f1_cutoff
            y_pred = np.hstack(output)>f1_cutoff
            auc_roc = roc_auc_score(test_y, np.hstack(output))
            print "LOG_AR_MS: F1 score: {:.2f}".format(lrb.f1_score(test_y, y_pred))
            print "LOG_AR_MS: F1 score: {:.2f} for 50pct cutoff".format(lrb.f1_score(test_y, np.hstack(output)>0.5))
            print "LOG_AR_MS: AUC score: {:.2f}".format(lrb.average_precision_score(test_y, np.hstack(output)))
            print "LOG_AR_MS: AUC(ROC) score: {:.2f}".format(auc_roc)
            print "LOG_AR_MS: Precision", sk.metrics.precision_score(test_y, y_pred)
            print "LOG_AR_MS: Recall", sk.metrics.recall_score(test_y, y_pred)
            print sk.metrics.confusion_matrix(test_y, y_pred)


def run_all_experiments():
    def apply_all_exp_deltas(config):
        config.n_epochs = 3
        return config

    # expname = dot_prod_nn_fulldataset
    expname = dot_prod_nn

    config = apply_all_exp_deltas(DotProdConfig())
    print "Running model 4"
    print config
    expname(config)

    del config
    config = apply_all_exp_deltas(DotProdConfig())
    config.model_three_prb = False
    print "Running model 3"
    print config
    expname(config)

    del config
    config = apply_all_exp_deltas(DotProdConfig())
    config.model_three = False
    print "Running model 2"
    print config
    expname(config)

    del config
    config = apply_all_exp_deltas(DotProdConfig())
    config.model_one = True
    print "Running model 1"
    print config
    expname(config)


def run_4(mod_fn, expname, *args):
    config = mod_fn(DotProdConfig(), *args)
    print "LOG_AR_MS: Running model 4"
    print "LOG_AR_MS: " + config.__repr__()
    expname(config)

    del config
    config = mod_fn(DotProdConfig(), *args)
    config.model_three_prb = False
    print "LOG_AR_MS: Running model 3"
    print "LOG_AR_MS: " + config.__repr__()
    expname(config)

    del config
    config = mod_fn(DotProdConfig(), *args)
    config.model_three = False
    print "LOG_AR_MS: Running model 2"
    print "LOG_AR_MS: " + config.__repr__()
    expname(config)

    del config
    config = mod_fn(DotProdConfig(), *args)
    config.model_one = True
    print "LOG_AR_MS: Running model 1"
    print "LOG_AR_MS: " + config.__repr__()
    expname(config)


def run_exps():
    expname = dot_prod_nn
    def mod1(config, n):
        config.n_epochs = n
        return config

    n_epoch_sets = [1, 2, 5, 10, 15, 25]
    for n_ep in n_epoch_sets:
        print "LOG_AR_MS: Running %d epoch set" % n_ep
        run_4(mod1, expname, n_ep)

    def mod2(config, lr):
        config.lr = lr
        return config

    learning_rates = [0.0001, 0.001, 0.0015, 0.005, 0.025, 0.05]
    for lr in learning_rates:
        print "LOG_AR_MS: Running %f learning rate" % lr
        run_4(mod2, expname, lr)

    def mod_do(config, dropout_rate):
        setattr(config, "dropout_rate", dropout_rate)
        return config

    dropout_rates = [0.2, 0.5, 0.8]
    for do in dropout_rates:
        print "LOG_AR_MS: Trying %.2f as a the dropout rate" % do
        run_4(mod_do, expname, do)

    def make_big(config, arch):
        config.hidden_size = arch[0]
        config.hidden_size2 = arch[1]
        config.hidden_size3 = arch[2]
        return config
    full_arch = [1000, 500, 500]
    print "LOG_AR_MS: Trying with larger architecture"
    run_4(make_big, expname, full_arch)

    # weighted NE loss wt range
    def mod_ne_ce_wt(config, new_wt):
        config.cross_entropy_weighted_poswt = new_wt
        return config

    other_wts = [1, 2, 5, 10, 20]
    for wt in other_wts:
        print "LOG_AR_MS: Trying with cross entropy wted val of %d" % wt
        run_4(mod_ne_ce_wt, expname, wt)

    # regularization range
    # @todo resolve issue with implementation on m2
    # def set_regularization(config, reg_val):
    #     setattr(config, "regularization_l2_weight", reg_val)
    #     return config
    #
    # l2_reg_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    # for l2_v in l2_reg_values:
    #     print "LOG_AR_MS: Trying with l2 regularization penalty of %f" % l2_v
    #     run_4(set_regularization, expname, l2_v)


def run_best_on_test():
    expname = dot_prod_nn_on_TEST
    dataset = "ALL"

    print "Running Model 4 on TEST"
    config = DotProdConfig()
    config.lr = 0.0001
    print config.__repr__()
    expname(config, dataset)
    del config

    print "Running Model 3 on TEST"
    config = DotProdConfig()
    config.model_three_prb = False
    config.hidden_size = 1000
    config.hidden_size2 = 500
    config.hidden_size3 = 500
    print config.__repr__()
    expname(config, dataset)
    del config

    print "Running Model 2 on TEST"
    config = DotProdConfig()
    config.model_three = False
    print config.__repr__()
    expname(config, dataset)
    del config

    print "Running Model 1 on TEST"
    config = DotProdConfig()
    config.model_one = True
    print config.__repr__()
    expname(config, dataset)


if __name__ == "__main__":
    # dot_prod_nn()
    # dot_prod_nn_fulldataset()
    # run_all_experiments()
    # run_exps()
    run_best_on_test()
