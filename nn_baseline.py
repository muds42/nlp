import pdb
import logging
import cPickle

import tensorflow as tf
import utils as cr_utils
import logistic_regression_baseline as lrb

import logging
import numpy as np

from q3_util import Progbar, minibatches

import argparse
import sys
import time
import logging

logger = logging.getLogger("very_simple_nn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def process_and_pickle_wordvecs():
    w2v_file_location = 'w2v/w2v_50d.txt'
    wordvec_dict = {line.split('\t')[0]: np.array(line.split('\t')[1].strip(' \n').split(' '), dtype='float')
                    for line in open(w2v_file_location)}
    utils.write_pickle(wordvec_dict, 'w2v/w2v.pkl')


def multi_word_function(input_arrays):
    # for now return a simple average
    # let's mark this as an area of potential improvement in the future
    return np.mean(input_arrays, axis=0)


def convert_train_examples(train_inp, wordvec_dict):
    ret_val = []
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
    for inp in train_inp:
        tmp = np.array([])
        for field in ALL_FIELDS:  # i am assuming this will enforce ordering
            if field not in inp.keys():
                tmp = np.append(tmp, np.array([0])) # this is to handle other sentence dist features
            elif field not in TEXT_FIELDS:
                tmp = np.append(tmp, np.array(inp[field]))
            else:
                words = inp[field].split(' ')
                tmp = np.append(tmp, multi_word_function(
                    [wordvec_dict.get(x, np.zeros(50)) for x in words]))
        ret_val.append(tmp)
    return ret_val


def prepare_data():
    train_x, train_y = lrb.build_examples("train")
    dev_x, dev_y = lrb.build_examples("dev")
    wordvec_dict = cr_utils.load_pickle('w2v/w2v.pkl')
    mod_train_x = convert_train_examples(train_x, wordvec_dict)
    mod_dev_x = convert_train_examples(dev_x, wordvec_dict)

    cr_utils.write_pickle((mod_train_x, np.array(train_y), mod_dev_x,
        np.array(dev_y)), "./w2v/nn_nominal_full_dataset.pkl")
    # create mini-training and eval sets for rapid iterations
    cr_utils.write_pickle((mod_train_x[:50000], np.array(train_y[:50000]),
        mod_dev_x[:10000], np.array(dev_y[:10000])), "./w2v/nn_nominal_mini_dataset.pkl")



class VerySimpleConfig:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    n_feat = 11*50 + 16  # 566 with the current enumeration
    hidden_size = 500
    hidden_size2 = 200
    # hidden_size3 = 100
    n_epochs = 25
    # lr = 0.001
    lr = 0.0025  # 2 has a f1 of 0.49 (LR is 0.39); 3 had f1 0.45
    batch_size = 300
    # dropout_rate = 0.5


class VerySimpleNN():
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_feat), name="x")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, ), name="y")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        x = self.inputs_placeholder
        cf = self.config
        W = tf.get_variable("W", shape=(cf.n_feat, cf.hidden_size),
            initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2", shape=(cf.hidden_size, cf.hidden_size2),
            initializer=tf.contrib.layers.xavier_initializer())

        b1 = tf.get_variable("b1", initializer=tf.zeros(cf.hidden_size, ))
        b2 = tf.get_variable("b2", initializer=tf.zeros(cf.hidden_size2, ))
        h = tf.nn.relu(tf.matmul(x, W) + b1)

        if hasattr(cf, 'dropout_rate'):
            hdrop = tf.nn.dropout(h, keep_prob = cf.dropout_rate)
            h2 = tf.nn.relu(tf.matmul(hdrop, W2) + b2)
        else:
            h2 = tf.nn.relu(tf.matmul(h, W2) + b2)
        # h2 = tf.nn.relu(tf.matmul(h, W2) + b2)

        if hasattr(cf, 'hidden_size3'):
            W3 = tf.get_variable("W3", shape=(cf.hidden_size2, cf.hidden_size3),
                initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", initializer=tf.zeros(cf.hidden_size3, ))
            ushape = cf.hidden_size3
            h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
        else:
            ushape = cf.hidden_size2

        U = tf.get_variable("U", shape=(ushape, 1),
            initializer=tf.contrib.layers.xavier_initializer())

        # hdrop = tf.nn.dropout(h2, keep_prob = cf.dropout_rate)  # this causes 0.63 -> 0.59
        # hdrop = tf.nn.dropout(h3, keep_prob = cf.dropout_rate)
        if hasattr(cf, 'hidden_size3'):
            pred = tf.sigmoid(tf.matmul(h3, U))
        else:
            pred = tf.sigmoid(tf.matmul(h2, U))

        return tf.transpose(pred) #state # preds

    def add_loss_op(self, preds):
        y = self.labels_placeholder
        loss = tf.nn.l2_loss(preds-y)
        # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(preds), reduction_indices=[1]))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data"""
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        # if inputs is None:
        #     inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        labels = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            # batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return preds

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            # import pdb; pdb.set_trace()
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])

        return losses

    def fit(self, sess, train):
        losses = []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss = self.run_epoch(sess, train)
            losses.append(loss)
            # grad_norms.append(grad_norm)

        return losses

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.grad_norm = None
        self.build()


def very_simple_nn():
    # datafile = './w2v/nn_mini_dataset.pkl'
    # datafile = './w2v/nn_full_dataset.pkl'
    datafile = "./w2v/nn_nominal_mini_dataset.pkl"
    datafile = "./w2v/nn_nominal_full_dataset.pkl"
    train_x, train_y, dev_x, dev_y = cr_utils.load_pickle(datafile)

    config = VerySimpleConfig()
    report = None

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = VerySimpleNN(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            # model.fit(session, [train_x, train_y])
            model.fit(session, zip(train_x, train_y))
            output = model.output(session, zip(dev_x))
            # score the predictions..
            # import pdb; pdb.set_trace()
            y_pred = np.floor(np.hstack(output)*2)  # 50% cutoff; maybe can be calibrated
            logger.info("\nSSE on predictions is: %f" % (np.sum(np.square(y_pred-dev_y))))
            print "F1 score: {:.2f}".format(lrb.f1_score(dev_y, y_pred))
            print "AUC score: {:.2f}".format(lrb.average_precision_score(dev_y, y_pred))

if __name__ == "__main__":
    very_simple_nn()
