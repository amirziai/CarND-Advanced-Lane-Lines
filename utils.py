import pickle as pickle_module


def unpickle(file_path):
    with open(file_path, 'rb') as file_handle:
        pickled_object = pickle_module.load(file_handle)
        return pickled_object


def pickle(object_to_pickle, file_path):
    with open(file_path, 'wb') as file_handle:
        pickle_module.dump(object_to_pickle, file_handle)
