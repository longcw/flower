import os


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if len(parent) > 0 and not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)