import os
import pickle

from constants import CACHE


## Caching parameters and data

def save_to_cache(namespace, key, data):
    """Save data/parameters to (reboot) persistent cache."""
    directory = os.path.join(CACHE, namespace)
    full_path = os.path.join(directory, "{}.pkl".format(key))
    os.makedirs(directory, exist_ok=True)

    with open(full_path, "wb") as fp:
        pickle.dump(data, fp)


def load_from_cache(namespace, key):
    """Load data/parameters from persistent cache. Returns None if not found."""
    directory = os.path.join(CACHE, namespace)
    full_path = os.path.join(directory, "{}.pkl".format(key))

    if not (os.path.exists(full_path)):
        return None

    with open(full_path, "rb") as fp:
        return pickle.load(fp)
