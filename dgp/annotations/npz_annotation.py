# Copyright 2022 Woven Planet. All rights reserved.
import os
from typing import Dict

import numpy as np

from dgp.annotations import Annotation
from dgp.utils.dataset_conversion import generate_uid_from_point_cloud


class NPZFeatureAnnotation(Annotation):
    """Container for generic numpy arrays.
    
    The intended use is for storing intermediate feature tensors as annotations. For example,
    Resnet features on images, CLIP embeddings, etc with particular focus on autolabels. If
    at any time a particular type of feature is found to be useful, it should be moved into
    its own annotation type.

    Parameters
    ----------
    depth: np.ndarray
        2D numpy float array that stores per-pixel depth.
    """
    def __init__(self, data: Dict[str, np.ndarray]):
        super().__init__(None)

        self._data = {}
        for k, v in data.items():
            assert isinstance(v, np.ndarray), f'key {k} should be a numpy array. got {type(v)}'
            self._data[k] = v

    @classmethod
    def load(cls, annotation_file, ontology=None):
        """Loads annotation from file into a canonical format for consumption in __getitem__ function in BaseDataset.

        Parameters
        ----------
        annotation_file: str
            Full path to NPZ file that stores the data

        ontology: None
            Dummy ontology argument to meet the usage in `BaseDataset.load_annotation()`.
        """
        assert ontology is None, "'ontology' must be 'None' for {}.".format(cls.__name__)
        data = np.load(annotation_file)
        data_dict = {k: v for k, v in data.items()}
        return cls(data_dict)

    @property
    def data(self):
        return self._data

    @property
    def hexdigest(self):
        # Hash each item
        keys = sorted(list(self.data.keys()))
        hashlets = [generate_uid_from_point_cloud(self.data[k]) for k in keys]
        hash_string = ''.join(hashlets)
        return generate_uid_from_point_cloud(hash_string.encode('utf-8'))

    def save(self, save_dir):
        """Serialize annotation object if possible, and saved to specified directory.
        Annotations are saved in format <save_dir>/<sha>.<ext>

        Paramaters
        ----------
        save_dir: str
            Path to directory to saved annotation

        Returns
        -------
        pointcloud_path: str
            Full path to the output NPZ file.
        """
        save_path = os.path.join(save_dir, '{}.npz'.format(self.hexdigest))
        np.savez_compressed(save_path, **self.data)
        return save_path

    def render(self):
        # NOTE: This is here because it is marked as abstract in Annotation
        pass
