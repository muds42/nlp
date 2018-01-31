import utils

import utils as cr_utils
import random
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

import logistic_regression_baseline as lrb
import custom_feature_append as cfa


# baseline load raw docs
def investigate():
    vv = utils.load_pickle('./data/processed_data/dev_nominals_filtered_mentions.pkl')
    docs = []
    for doc in vv:
        docs.append(doc)
    return docs



def build_examples(dataset, downsample_prob=0.02):
    print "Building examples for dataset", dataset
    xs, ys, zs = [], [], []
    filtered_mentions = utils.load_pickle(
        # './data/processed_data/' + dataset + '_filtered_mentions.pkl')
        './data/processed_data/' + dataset + '_nominals_filtered_mentions.pkl')
    for doc_id, doc in enumerate(utils.load_json_lines("./data/raw_data/" + dataset)):
        for key in filtered_mentions[doc_id]:
            y = doc["labels"][key]
            # downsample negative examples to help with class imbalance and keep things running fast
            if y == 1 or random.random() < downsample_prob:
                xs.append(lrb.featurize_mentions(key.split(), doc))
                ys.append(y)
                zs.append(featurize_mentions2(key.split(), doc))
    return xs, ys, zs


def featurize_distance(d, prefix):
    features = {}
    if d < 5:
        features["distance = " + str(d)] = 1
    else:
        features["distance > 5"] = 1
    features["distance"] = min(d, 80) / float(80)
    return {prefix + k: v for k, v in features.iteritems()}


def featurize_mentions2((mention_num_1, mention_num_2), doc):
    m1, m2 = doc["mentions"][mention_num_1], doc["mentions"][mention_num_2]

    features = {"bias": 1}
    for i, feature_name in enumerate(doc["pair_feature_names"]):
        features[feature_name] = doc["pair_features"][mention_num_1 + " " + mention_num_2][i]

    features.update(featurize_distance(m2["sent_num"] - m1["sent_num"], "sentence "))
    features.update(featurize_distance(m2["mention_num"] - m1["mention_num"], "mention "))

    for m, prefix in [(m1, "m1 "), (m2, "m2 ")]:
        sentence = m["sentence"]
        start, end, head = m["start_index"], m["end_index"], m["head_index"]
        get_word = lambda i: sentence[i].lower() if 0 < i < len(sentence) else "NA"
        features[prefix + "prev word"] = get_word(start - 1)
        features[prefix + "first word"] = get_word(start)
        features[prefix + "last word"] = get_word(end - 1)
        features[prefix + "next word"] = get_word(end)
        features[prefix + "head word"] = get_word(head)
        features[prefix + "full mention"] = " ".join(sentence[start:end])
    features["heads"] = m1["sentence"][m1["head_index"]] + "_" + m2["sentence"][m2["head_index"]]

    return features


if __name__ == '__main__':
    dev_x, dev_y, dev_z = build_examples("dev")
    wordvec_dict = cr_utils.load_pickle('w2v/w2v.pkl')
    mod_dev_x = cfa.convert_train_examples(dev_x, wordvec_dict)

    cr_utils.write_pickle((mod_dev_x, np.array(dev_y), dev_z),
        "./w2v/nn_nominal_downsample_WERR.pkl")
