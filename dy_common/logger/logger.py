import sys


class Logger(object):
    """
    A quick class which allows stdout to be piped to a file in addition to the terminal
    """

    def __init__(self, filename, filemode):
        self.terminal = sys.stdout
        self.log = open(filename, filemode)

    def __del__(self):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def set_logger(filename, filemode='w'):
    """
    Pipe stdout to the terminal in addition to a file.
    """
    assert filemode in ['a', 'w'], 'set_logger: filemode must be \'a\' or \'w\''

    sys.stdout = Logger(filename, filemode)
