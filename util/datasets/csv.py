"""
Module to parse csv dataset files.
"""

import numpy as np
import os


class CSVParser:
    """
    Parses a file containing datasets paths and labels.

    Expected format:
        'name_of_file_1.ext', ..., ..., 'label_1'
        'name_of_file_2.ext', ..., ..., 'label_2'
        ...
        'name_of_file_n.ext', ..., ..., 'label_n'

        *ext is the file extension.

    :param path: str
        Path to the file to parse.
    """
    def __init__(self, path: str):
        self.path = path
        self.file_names = []
        self.labels = []
        with open(self.path) as dataset:
            for line in dataset:
                data = line.split(',')
                n = data[0]
                if os.sep == '\\':
                    n = n.replace('/', os.sep)
                elif os.sep == '/':
                    n = n.replace('\\', os.sep)
                self.file_names.append(n)
                self.labels.append(data[-1].rstrip())

    def data_and_labels(self) -> (list, list):
        """
        Returns the parsed datasets and labels

        :return: tuple
            A list with the file_names and a list with the respective labels of
            each instance.
        """
        return self.file_names, self.labels

    def __call__(self, *args, **kwargs):
        """Short hand for data_and_labels method"""
        return self.data_and_labels()

    @property
    def num_classes(self):
        return len(np.unique(self.labels))
