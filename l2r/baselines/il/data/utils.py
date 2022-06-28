import re


def tryint(s):
    try:
        return int(s)
    except Exception:
        return s

"""
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
"""

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(lst):
    """ Sort the given list in the way that humans expect.
    """
    lst.sort(key=alphanum_key)
