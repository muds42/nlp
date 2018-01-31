import utils
import random
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score


def build_examples(dataset, downsample_prob=0.02):
    print "Building examples for dataset", dataset
    xs, ys = [], []
    filtered_mentions = utils.load_pickle(
        # './data/processed_data/' + dataset + '_filtered_mentions.pkl')
        './data/processed_data/' + dataset + '_nominals_filtered_mentions.pkl')
    for doc_id, doc in enumerate(utils.load_json_lines("./data/raw_data/" + dataset)):
        for key in filtered_mentions[doc_id]:
            y = doc["labels"][key]
            # downsample negative examples to help with class imbalance and keep things running fast
            if y == 1 or random.random() < downsample_prob:
                xs.append(featurize_mentions(key.split(), doc))
                ys.append(y)
    return xs, ys


def featurize_distance(d, prefix):
    features = {}
    if d < 5:
        features["distance = " + str(d)] = 1
    else:
        features["distance > 5"] = 1
    features["distance"] = min(d, 80) / float(80)
    return {prefix + k: v for k, v in features.iteritems()}


def featurize_mentions((mention_num_1, mention_num_2), doc):
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
    features["heads"] = m1["sentence"][m1["head_index"]] + "_" + m2["sentence"][m2["head_index"]]

    return features


def preprocess_data():
    xs_train, ys_train = build_examples("train")
    xs_dev, ys_dev = build_examples("dev")
    print "Vectorizing and writing examples"
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(xs_train)
    X_dev = vectorizer.transform(xs_dev)
    utils.write_pickle((X_train, np.array(ys_train), X_dev, np.array(ys_dev),
                        # vectorizer.vocabulary_), "./data/processed_data/LR_dataset.pkl")
                        vectorizer.vocabulary_), "./data/processed_data/LR_nominals_dataset.pkl")


def run_model():
    print "Loading examples"
    X_train, y_train, X_dev, y_dev, vocab = utils.load_pickle(
        # "./data/processed_data/LR_dataset.pkl")
        "./data/processed_data/LR_nominals_dataset.pkl")

    print "Fitting model"
    clf = LogisticRegression(C=10)
    clf.fit(X_train, y_train)

    print "Top feature weights:"
    feature_weights = sorted([(k, clf.coef_[0, i]) for k, i in vocab.items()],
                             key=lambda (k, s): abs(s), reverse=True)
    for k, v in feature_weights[:20]:
        print "  {:}, {:.2f}".format(k, v)

    print "Evaluating model"
    y_probs = clf.predict_proba(X_dev)[:, 1]
    y_pred = np.floor(y_probs * 2)
    print "F1 score: {:.2f}".format(f1_score(y_dev, y_pred))
    print "AUC score: {:.2f}".format(average_precision_score(y_dev, y_probs))


def main():
    random.seed(0)
    preprocess_data()
    run_model()

if __name__ == '__main__':
    main()
