import os
from typing import Iterable, Optional

class cumulative:
    
    """A cumulative iterator."""
    
    def __init__(self, iterable: Iterable, start: Optional[int] = 0):
        self.iterable = iter(iterable)
        self.sum = start
        return None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = next(self.iterable)
        I = len(item)
        self.sum += I
        return (self.sum - I, item)        



def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


