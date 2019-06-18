"""
This module implements a data generator utility based on the
keras.utils.Sequence

A good usage of this module is in the fit_generator method of Keras library.
"""

from abc import abstractmethod
import keras
import random
import numpy


class Generator(keras.utils.Sequence):
    """A generator which provides batches of data"""

    def __init__(self, paths, labels, batch_size: int,
                 loader_fn: callable = None, pre_process_fn: callable = None,
                 shuffle: bool = True, expected_shape: tuple=None,
                 not_found_ok=False, **loader_kw):
        """
        Initializes a generator.

        Is is necessary to provide or implement a loader function which loads
        data. The pre processing function is optional, and will be executed
        always before returning a new batch of data.

        :param paths:
            List containing the data file paths to load.

        :param labels:
            List containing the labels of each data.

        :param batch_size: int
            The size of the batch to queue.

        :param loader_fn: callable(source_path: str)
            A function to load the data, if None, must override the loader
            function.

        :param pre_process_fn: callable(data) -> new_data
            A function to pre process the data after it is loaded, and before
            returning the batch. Optional.

        :param shuffle: bool
            If true, shuffle the paths before loading them.

        :param expected_shape: tuple
            Check if the shape of each loaded data is in a proper format. If
            not, the data will be ignored.

        :param not_found_ok: bool
            If false, will raise a FileNotFoundError, if true,  will ignore
            not found files. Default to false.

        :param loader_kw: Additional kwargs to be passed on to the loader
            function.

        Note: if a expected_shape is provided or not_found_ok is True, the
        __getitem__ method will load a random instance to avoid raising
        exceptions.
        """

        self._paths = paths
        self._labels = labels
        self._pre_process_fn = pre_process_fn
        self._batch_size = batch_size
        self._loaderkw = loader_kw
        self._not_found_ok = not_found_ok
        self._expected_shape = expected_shape
        if loader_fn is not None:
            self.loader = loader_fn

        if shuffle:
            dataset = list(zip(self._paths, self._labels))
            random.shuffle(dataset)
            self._paths, self._labels = zip(*dataset)

    @abstractmethod
    def loader(self, source_path: str, *args, **kwargs):
        """
        A loader of data. Must implement a loader for correct operation of the
        buffer.

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

    def _get_random_instance(self):
        i = numpy.random.randint(0, len(self._paths))
        return self._paths[i], self._labels[i]

    def __getitem__(self, index) -> (numpy.ndarray, numpy.ndarray):
        """
        Gets a batch of data.

        :return: (numpy.ndarray, numpy.ndarray)
            Tuple of arrays. The first array represents the data, and the
            second represents the labels.

        Note: in case the size of the batch is greater than the amount of
        data available, only the read amount will be returned.
        """
        paths = self._paths[(index*self._batch_size):
                            ((index+1)*self._batch_size)]
        labels = self._labels[(index*self._batch_size):
                              ((index+1)*self._batch_size)]
        paths_and_labels = list(zip(paths, labels))
        # Fill batches
        x = []
        y = []
        threshold = 0
        for path_label in paths_and_labels:
            if self._not_found_ok:
                try:
                    # Try to load the data
                    x.append(self.loader(path_label[0], **self._loaderkw))
                    y.append(path_label[1])
                except FileNotFoundError:
                    # If not found, append a new path to load
                    p, l = self._get_random_instance()
                    paths_and_labels.append((p, l))
                    # Increase a threshold value to avoid infinite loops
                    threshold += 1

                    # If all data was tried to be read, raise an exception
                    if threshold == self._batch_size:
                        # (threshold can be any value)
                        raise RuntimeError(
                            'Threshold value reached. Error when '
                            'trying to read the files provided '
                            '(not able to fill the batch).')
                    continue
            else:  # Read data without handling the exception
                y.append(path_label[1])
                x.append(self.loader(path_label[0], **self._loaderkw))

            if self._expected_shape is not None and x[-1].shape != \
                    self._expected_shape:
                # If the last read data is not in the expected shape
                p, l = self._get_random_instance()
                paths_and_labels.append((p, l))
                # Increase a threshold value to avoid infinite loops
                threshold += 1
                # Remove the last instance
                x.pop()
                y.pop()

                # If all data was tried to be read, raise an exception
                if threshold == self._batch_size:
                    raise RuntimeError('Threshold value reached. Error when '
                                       'trying to read the files provided '
                                       '(not able to fill the batch).')
                continue

        if self._pre_process_fn is not None:
            x = self._pre_process_fn(x)

        return numpy.asarray(x), numpy.asarray(y)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(numpy.floor(len(self.paths) / self._batch_size))

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

    def on_epoch_end(self):
        pass
