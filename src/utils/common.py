import os

WORKSPACE = os.path.join(os.path.dirname(__file__), "../..")


def get_absolute_path(path_):
    if not os.path.isabs(path_):
        return os.path.join(WORKSPACE, path_)
    else:
        return path_
