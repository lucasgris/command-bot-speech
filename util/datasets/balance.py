"""
This module provide functions to balance data.
"""
import numpy


def balance_data(x, y):
    """
    Balance data to be representative.

    Note: When balancing the data, the number os data and labels tends to
    decrease because the extra instances of some classes will be removed.

    :param x:
        Data or paths.
    :param y:
        Labels/classes.

    :return: tuple (numpy.ndarray, numpy.ndarray)
        x, y balanced
    """
    dataset = list(zip(x, y))
    unique, counts = numpy.unique(y, return_counts=True)
    max_n_instances = min(counts)
    paths_per_label = dict()
    for label in unique:
        paths_per_label[label] = list(filter(lambda d: d[1] == label, dataset))
    x = []
    y = []
    for label in paths_per_label:
        for inst, lbl in paths_per_label[label][:max_n_instances]:
            x.append(inst)
            y.append(lbl)

    return numpy.asarray(x), numpy.asarray(y)
