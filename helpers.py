import pickle


def import_pickle(path):
    """
    Load and return a Python object from a pickle file.
    """
    with open(path, "rb") as pkl_file:
        file = pickle.load(pkl_file)
    return file


def save_pickle(path, object):
    """
    Save a Python object to a file using pickle.
    """
    with open(path, "wb") as pkl_file:
        pickle.dump(object, pkl_file)


def increment_or_add(dict_obj, key):
    """
    Increment the value of a dictionary key, or add the key with value 1 if it doesn't exist.
    """
    try:
        dict_obj[key] += 1
    except KeyError:
        dict_obj[key] = 1


def generate_all_subpaths(path, lengths=None):
    """
    Generate all subpaths of a given path with specific lengths.
    """
    subpaths = []
    n = len(path)
    if lengths == None:
        lengths_to_get = range(2, n + 1)
    else:
        lengths_to_get = lengths
    for length in lengths_to_get:
        subpaths.extend(tuple(path[i : i + length]) for i in range(n - length + 1))
    return subpaths
