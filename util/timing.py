import time


def timing(c: callable) -> callable:
    """
    Outputs the time a function a callable object takes to execute.

    :param c: callable
        Callable object.
    :return: function
        Wrapper function to the callable object, which measures the elapsed
        time.
    """
    def wrapper(*args, **kwargs):
        """
        :param args: arguments are passed through to the callable object.
        :param kwargs: K arguments are passed through to the callable object.
        """
        t1 = time.time()
        c(*args, **kwargs)
        t2 = time.time()
        print('%r (%r, %r) %2.2f sec' % (c.__name__, args, kwargs, t2 - t1))

    return wrapper


class Timer:
    def __enter__(self):
        self._start = time.time()
        self._interval = None
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._interval = self._end - self._start

    @property
    def interval(self):
        return self._interval
