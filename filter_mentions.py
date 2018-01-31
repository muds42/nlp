import utils
from collections import defaultdict


def write_filtered_mentions(dataset):
    types_to_keep = ("NOMINAL")  # "PROPER" you might even want to start with only nominals

    filtered_mentions = defaultdict(list)
    for doc_id, doc in enumerate(utils.load_json_lines("./data/raw_data/" + dataset)):
        head_match_index = doc["pair_feature_names"].index("relaxed-head-match")
        for key in doc["labels"]:
            mention_num_1, mention_num_2 = key.split()
            m1, m2 = doc["mentions"][mention_num_1], doc["mentions"][mention_num_2]
            if doc["pair_features"][key][head_match_index] == 0 and m1["mention_type"] \
                    in types_to_keep and m2["mention_type"] in types_to_keep:
                filtered_mentions[doc_id].append(key)

    print "dataset size", sum(len(ms) for ms in filtered_mentions.values())
    utils.write_pickle(filtered_mentions,
                       './data/processed_data/' + dataset + '_nominals_filtered_mentions.pkl')
                    #    './data/processed_data/' + dataset + '_filtered_mentions.pkl')


def main():
    write_filtered_mentions("train")
    write_filtered_mentions("dev")
    write_filtered_mentions("test")

if __name__ == '__main__':
    main()
