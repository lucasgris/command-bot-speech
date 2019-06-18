"""
This script creates a csv file containing data paths and labels to process.
"""
import argparse
import random
import glob
import os
import numpy as np
from util.datasets import balance
from collections import defaultdict


def validate_args(paths_and_labels, data_balance, shuffle, test_split,
                  val_split):
    if len(paths_and_labels) % 2 != 0:
        raise ValueError('Incorrect number of paths and labels={}'.
                         format(paths))
    for p in paths:
        if not os.path.isdir(p):
            raise ValueError('Not a valid directory: {}'.format(p))
    if not data_balance and test_split > 0.0:
        print('[WARN] Testing split is set but data balancing will not be '
              'performed.')
    if not data_balance and val_split > 0.0:
        print('[WARN] Validation split is set but data balancing will not be '
              'performed.')
    if shuffle and test_split > 0.0:
        print('[WARN] Testing split is set but data shuffling will not be '
              'performed.')
    if val_split + test_split > 1.0:
        raise ValueError('Incorrect split proportion: validation proportion={},'
                         ' test proportion={} and train proportion={}'.
                         format(val_split, test_split,
                                1 - (val_split + test_split)))


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a CSV file '
                                                 'containing data paths and '
                                                 'labels to process.')
    parser.add_argument('pl', help='Paths and labels. First provide the path '
                                   'where the data is, and after each path '
                                   'provide the respective label.'
                                   'For instance, path/to/cats/images cats '
                                   'path/to/dogs/images dogs ...',
                        nargs='+')
    parser.add_argument('f', help='Output path to the CSV file.')
    parser.add_argument('--shuffle', help='Shuffles the paths and labels',
                        action='store_true', default=False)
    parser.add_argument('--balance', help='Balances data to be representative. '
                                          'Note: some files will be ignored '
                                          'while balancing the number of '
                                          'instances of each class.',
                        action='store_true', default=False)
    parser.add_argument('--test_split', help='Creates another two CSV files '
                                             'and split data paths and labels '
                                             'into two data sets for training '
                                             'and testing. Provide a number '
                                             'between 0.0 and 1.0 that '
                                             'represents the proportion of the '
                                             'dataset to split.',
                        type=float, default=0.0)
    parser.add_argument('--val_split', help='Creates another two CSV files '
                                            'and split data paths and labels '
                                            'into two data sets for training '
                                            'and testing. Provide a number '
                                            'between 0.0 and 1.0 that '
                                            'represents the proportion of the '
                                            'dataset to split.',
                        type=float, default=0.0)
    parser.add_argument('--format', help='Format of the data.',
                        default='*')
    parser.add_argument('--r', help='Get paths recursively.',
                        action='store_true', default=False)

    args = parser.parse_args()

    paths_and_labels = list(map(lambda pl: pl.replace(os.sep, '/'), args.pl))
    csv_path = str(args.f).replace(os.sep, '/')
    recursive = args.r
    data_format = args.format
    shuffle = args.shuffle
    test_split = args.test_split
    val_split = args.val_split
    data_balance = args.balance
    paths = paths_and_labels[0::2]
    labels = paths_and_labels[1::2]

    validate_args(paths_and_labels, data_balance, shuffle, test_split,
                  val_split)

    splits = [0, 1 - (val_split + test_split), 1 - test_split]

    dataset = list()

    for path, label in zip(paths, labels):
        if recursive:
            file_paths = glob.glob(path + '/**/*.' + data_format,
                                   recursive=True)
        else:
            file_paths = glob.glob(path + '/*.' + data_format)
        if len(file_paths) == 0:
            print('[ERROR] Not found. Check the path and format provided. '
                  'Arguments provided. Not found for {path} and {label}'.
                  format(path=path, label=label))
            exit(-1)
            for k, v in args:
                print(k, ':', v)
            continue

        for file_path in file_paths:
            dataset.append((file_path, label))

    if shuffle:
        random.shuffle(dataset)

    file_paths, file_labels = zip(*dataset)
    num_classes = np.unique(file_labels)

    if balance:
        file_paths, file_labels = balance.balance_data(file_paths, file_labels)
    else:
        file_paths = np.asarray(file_paths)
        file_labels = np.asarray(file_labels)

    with open(csv_path, 'w') as csv_file:
        print('[INFO] creating a csv file with paths and labels')
        for p, l in zip(file_paths, file_labels):
            csv_file.write(p + ',' + l + '\n')
    if test_split > 0.0 or val_split > 0.0:
        paths_per_label = defaultdict(lambda: [])
        for p, l in zip(file_paths, file_labels):
            paths_per_label[l].append(p)
        train_paths = list()
        train_labels = list()
        val_paths = list()
        val_labels = list()
        test_paths = list()
        test_labels = list()
        for label in paths_per_label:
            train_paths += paths_per_label[label][splits[0]:int(splits[1]*len(
                paths_per_label[label]))]
            train_labels += [label for _ in range(splits[0], int(len(
                paths_per_label[label]) - splits[1] * len(
                paths_per_label[label])))]
            if val_split > 0.0:
                val_paths += paths_per_label[label][int(splits[1]*len(
                    paths_per_label[label])):int(splits[2]*len(
                    paths_per_label[label]))]
                val_labels += [label for _ in range(0, int(splits[1] * len(
                    paths_per_label[label]) - splits[2] * len(
                    paths_per_label[label])))]
            if test_split > 0.0:
                test_paths += paths_per_label[label][int(splits[2]*len(
                    paths_per_label[label])):len(paths_per_label[label])]
                test_labels += [label for _ in range(0, int(splits[2] * len(
                    paths_per_label[label])))]

        with open(csv_path[:-4] + '_train_data.csv', 'w') as csv_file:
            print('[INFO] creating a csv file with train data, '
                  'total', len(train_paths))
            for p, l in zip(train_paths, train_labels):
                csv_file.write(p + ',' + l + '\n')

        with open(csv_path[:-4] + '_val_data.csv', 'w') as csv_file:
            print('[INFO] creating a csv file with validation data, '
                  'total', len(val_paths))
            for p, l in zip(val_paths, val_labels):
                csv_file.write(p + ',' + l + '\n')

        with open(csv_path[:-4] + '_test_data.csv', 'w') as csv_file:
            print('[INFO] creating a csv file with test data, '
                  'total', len(test_paths))
            for p, l in zip(test_paths, test_labels):
                csv_file.write(p + ',' + l + '\n')
