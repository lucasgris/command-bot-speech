"""
Exceptions package

>>> try:
...     raise NotLoadedError("This is a test message")
... except NotLoadedError as e:
...     print(e)
This is a test message
"""


class NotLoadedError(Exception):
    """Exception for not loaded files when trying to perform some operation"""
    def __init__(self, message):
        super(NotLoadedError, self).__init__(message)
