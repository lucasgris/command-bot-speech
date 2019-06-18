from abc import abstractmethod
import random


class BaseDataLoader:
    def __init__(self, paths, labels, loader_fn: callable = None,
                 pre_process_fn: callable = None, shuffle: bool=True,
                 **loader_kw):
        if len(paths) != len(labels):
            raise ValueError('Paths and labels must have the same number of '
                             'instances.')
        self._pre_process_fn = pre_process_fn
        if shuffle:
            paths_labels = list(zip(paths, labels))
            random.shuffle(paths_labels)
            paths, labels = zip(*paths_labels)
        self._paths = paths
        self._labels = labels
        if loader_fn is not None:
            self.loader = loader_fn
        self._loaderkw = loader_kw

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError('Must implement a method to get trainable '
                                  'data')

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError('Must implement a method to get testable '
                                  'data')

    @abstractmethod
    def loader(self, source_path: str, *args, **kwargs):
        """
        A loader of data. Must implement a loader for correct operation of the
        dataloader.

        A loader accepts the source path of the file and returns the read data.

        :param source_path: str
            The path of the source to load.
        :param args: (optional)
            Additional args can be passed on to the loader function.
        :param kwargs: (optional)
            Additional kwargs can be passed on to internal function behavior.
        :return:
            The read object.
        """
        raise NotImplementedError('Loader not implemented. Must implement '
                                  'a loader for correct operation.')

    @property
    def size(self) -> int:
        """Returns the total quantity of original data"""
        return len(self._paths)

    @property
    def paths(self) -> list:
        """Returns a list containing the paths of data files"""
        return self._paths

    @property
    def labels(self):
        """Returns a list containing the labels of each instance of data"""
        return self._labels
