import pandas as pd
from modules import tqdm
import argparse
import codecs
import os


def pythainer_preprocess_pos(
        data_dir, train_name="train_pos.txt", dev_name="test_pos.txt", test_name="test_pos.txt"):
    train_f = read_data_pos(os.path.join(data_dir, train_name))
    dev_f = read_data_pos(os.path.join(data_dir, dev_name))
    test_f = read_data_pos(os.path.join(data_dir, test_name))

    train = pd.DataFrame({"labels": [x[0] for x in train_f], "text": [x[1] for x in train_f], "pos": [x[2] for x in train_f] })
    train["cls"] = train["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    train.to_csv(os.path.join(data_dir, "{}.train_pos.csv".format(train_name)), index=False, sep="\t")

    dev = pd.DataFrame({"labels": [x[0] for x in dev_f], "text": [x[1] for x in dev_f], "pos": [x[2] for x in dev_f]})
    dev["cls"] = dev["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    dev.to_csv(os.path.join(data_dir, "{}.dev_pos.csv".format(dev_name)), index=False, sep="\t")

    test_ = pd.DataFrame({"labels": [x[0] for x in test_f], "text": [x[1] for x in test_f], "pos": [x[2] for x in test_f]})
    test_["cls"] = test_["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    test_.to_csv(os.path.join(data_dir, "{}.test_pos.csv".format(test_name)), index=False, sep="\t")


def read_data_pos(input_file):
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        poses = []
        f_lines = f.readlines()

    # i is for test 
        i = 0
        for line in tqdm(f_lines, total=len(f_lines), desc="Process {}".format(input_file)):
            contends = line.strip()

            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue

            # this is Not used for Thai data
            if len(contends) == 0 and not len(words):
                words.append("")

            # this is blank-line as ending of sentence
            # also check labels > 0 so that it is not blank line
            if len(contends) == 0 and len(labels) > 0:
                lbl = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                p = ' '.join([pos for pos in poses if len(pos) > 0])
                lines.append([lbl, w, p])
                words = []
                labels = []
                poses = []
                continue

            if len(line.strip().split(' ')) >= 2:
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                pos = line.strip().split(' ')[2]
                words.append(word)
                labels.append(label.replace("-", "_"))
                poses.append(pos)

    #         if (i < 10):
    #             print(lbl)
    #             print(w)

    #         i += 1
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_name', type=str, default="train.txt")
    parser.add_argument('--dev_name', type=str, default="test.txt")
    parser.add_argument('--test_name', type=str, default="test.txt")
    return vars(parser.parse_args())


if __name__ == "__main__":
    pythainer_preprocess_pos(**parse_args())
