
"""
Some simple ANSI color functions.
"""

try:
    import colorama  # allow Windows to use ANSI colors
    colorama.init()
except:
    pass


RED   = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def red(s):
    "Return a red version of a string"
    return RED + s + RESET

def green(s):
    "Return a green version of a string"
    return GREEN + s + RESET

def redgreen(v, fmt="{:.2f}", sign=1):
    """
    Return a red or green version of a number with the given format.
    If sign is 1, negative values will be red - if sign is -1, they'll be green.
    """
    s = fmt.format(v)
    return red(s) if (v*sign)<0 else green(s)


