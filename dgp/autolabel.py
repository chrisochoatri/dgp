import logging
import os
from collections import deque
from copy import deepcopy
from shutil import copy2
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pyquaternion import Quaternion
from tqdm import tqdm

from dgp import (
    AUTOLABEL_FOLDER, AUTOLABEL_SCENE_JSON_NAME, BOUNDING_BOX_2D_FOLDER, BOUNDING_BOX_3D_FOLDER, DEPTH_FOLDER,
    INSTANCE_SEGMENTATION_2D_FOLDER, INSTANCE_SEGMENTATION_3D_FOLDER, ONTOLOGY_FOLDER, SEMANTIC_SEGMENTATION_2D_FOLDER,
    SEMANTIC_SEGMENTATION_3D_FOLDER
)
from dgp.annotations import (ANNOTATION_REGISTRY, ONTOLOGY_REGISTRY, Annotation, Ontology)
from dgp.annotations.bounding_box_3d_annotation import (BoundingBox3D, BoundingBox3DAnnotationList)
from dgp.constants import ANNOTATION_KEY_TO_TYPE_ID, ANNOTATION_TYPE_ID_TO_KEY
from dgp.proto.scene_pb2 import Scene
from dgp.utils.pose import Pose
from dgp.utils.protobuf import open_pbobject, save_pbobject_as_json
from mmdet3d.core.bbox import LiDARInstance3DBoxes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: create constants for all the other annotation types, missing keypoints, agent normals etc
ANNOTATION_TYPE_ID_TO_FOLDER = {
    'bounding_box_2d': BOUNDING_BOX_2D_FOLDER,
    'bounding_box_3d': BOUNDING_BOX_3D_FOLDER,
    'depth': DEPTH_FOLDER,
    'semantic_segmentation_2d': SEMANTIC_SEGMENTATION_2D_FOLDER,
    'semantic_segmentation_3d': SEMANTIC_SEGMENTATION_3D_FOLDER,
    'instance_segmentation_2d': INSTANCE_SEGMENTATION_2D_FOLDER,
    'instance_segmentation_3d': INSTANCE_SEGMENTATION_3D_FOLDER,
}


class DGPAutolabelScene():
    def __init__(self, dataset, scene_index, model_name, autolabel_keys: List[str], ontology_table=None):

        self.scene_index = scene_index
        self.dataset_item_index = dataset.dataset_item_index
        self.datum_index = dataset.scenes[scene_index].datum_index
        base_scene_file = dataset.scenes[scene_index].scene_path
        self.scene = open_pbobject(base_scene_file, Scene)
        self.scene_dir = os.path.dirname(base_scene_file)
        self.model_name = model_name
        self.autolabel_keys = autolabel_keys

        if ontology_table is None:
            ontology_table = dict()

        self.ontology_table = ontology_table

        # If an ontology is not supplied, try and copy the ontology from the base scene

        # TODO: switch this to a per key comparison
        for ontology_type_id, ontology_hash in self.scene.ontologies.items():
            annotation_type = ANNOTATION_TYPE_ID_TO_KEY[ontology_type_id]
            if annotation_type in self.autolabel_keys and annotation_type not in self.ontology_table:
                logging.warning(f'No ontology table supplied for {annotation_type} copying ontology from base dataset')
                ontology_file = os.path.join(self.scene_dir, ONTOLOGY_FOLDER, ontology_hash + '.json')
                self.ontology_table[annotation_type] = ONTOLOGY_REGISTRY[annotation_type].load(ontology_file)

    def save_ontology(self):
        """Saves the ontology files. If no ontology_table was passed, attemps to copy the required ontologies
        from the base scene"""
        ontology_dir = os.path.join(
            self.scene_dir,
            AUTOLABEL_FOLDER,
            self.model_name,
            ONTOLOGY_FOLDER,
        )
        os.makedirs(ontology_dir, exist_ok=True)

        for annotation_type in self.autolabel_keys:
            if annotation_type in self.ontology_table:
                ontology_type_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_type]
                ontology_path = self.ontology_table[annotation_type].save(ontology_dir)
                self.scene.ontologies[ontology_type_id] = os.path.basename(ontology_path).replace('.json', '')
                print(
                    'saved ontology', ontology_type_id,
                    os.path.basename(ontology_path).replace('.json', ''), ontology_dir
                )

    def save_annotation(self, idx: int, annotation: Annotation, annotation_key: str) -> None:
        """Save a dgp annotation object"""

        #TODO verify save paths per annotation/datum type, i.e DDAD has depth as
        # depth/lidar/camera_01 etc but there is no reference to this structure anywhere in
        # dgp.

        # TODO: add an autolabel root so we can save files elsewhere

        assert isinstance(annotation, ANNOTATION_REGISTRY[annotation_key])

        annotation_folder = ANNOTATION_TYPE_ID_TO_FOLDER.get(annotation_key, annotation_key)
        autolabel_dir = os.path.join(self.model_name, annotation_folder)
        save_dir = os.path.join(self.scene_dir, AUTOLABEL_FOLDER, autolabel_dir)
        os.makedirs(save_dir, exist_ok=True)

        datum = self.scene.data[idx].datum
        datum_type = datum.WhichOneof('datum_oneof')
        datum_value = getattr(datum, datum_type)  # This is datum.image or datum.point_cloud etc

        annotation_type_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_key]

        annotation_path = annotation.save(save_dir)
        datum_value.annotations[annotation_type_id] = os.path.join(autolabel_dir, os.path.basename(annotation_path))

    def save_sample(self, idx: int, sample: List[Dict[str, Any]]) -> None:
        """Save a dgp annotation sample that contains the requested annotation keys"""
        scene_index, sample_idx_in_scene, _ = self.dataset_item_index[idx]
        assert scene_index == self.scene_index

        for datum in sample:
            datum_name = datum['datum_name']
            for datum_key in datum:
                if datum_key not in self.autolabel_keys:
                    continue
                datum_idx_in_scene = self.datum_index[sample_idx_in_scene].loc[datum_name.lower()].data
                self.save_annotation(datum_idx_in_scene, datum[datum_key], datum_key)

    def save(
        self,
        dataset_idx: int,
        samples_pred: List[List[Dict[str, Any]]],
    ) -> None:
        """Saves a new autolabel scene file, annotations, and ontology"""

        save_dir = os.path.join(self.scene_dir, AUTOLABEL_FOLDER, self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        for idx, sample in zip(dataset_idx, samples_pred):
            self.save_sample(idx, sample)

        self.save_ontology()

        # TODO: should we attempt to zero out/blank out references to source data in the
        # new scene.json?
        save_pbobject_as_json(self.scene, os.path.join(save_dir, AUTOLABEL_SCENE_JSON_NAME))


class DGPAutoLabeler():
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        model_name: str,
        autolabel_keys: List[str],
        post_processor: Optional[Callable] = None,
        ar_steps: int = 1,
        ontology_table: Optional[Dict[str, Ontology]] = None
    ):
        self.model = model
        self.ar_steps = ar_steps
        self.autolabel_keys = autolabel_keys

        if post_processor is None:
            post_processor = lambda x: x
        self.post_processor = post_processor

        # if model_name is None:
        #     # TODO: replace with some model hash
        #     model_name = 'model_' + ''.join([str(x) for x in np.random.choice(9, 16, replace=True)])
        self.model_name = model_name

        self.ontology_table = ontology_table

    def predict(self, batch_sample, outputs):
        # If you need to pass prior outputs (hidden states, previous samples etc)
        # then this is where you modify the call signature of your model
        return self.model.forward(batch_sample)

    def preprocess_dgp_sample(self, samples: List[List[Dict[str, Any]]], device):
        # whatever you need to do to format dgp sample input for your model goes here
        # the input here is a list of samples but the output should just be the conversion
        # for the last sample. The output of this is whatever your model expects as input
        # not withstanding hidden states
        return samples[-1]

    def postprocess_to_dgp(self, sample, predictions_raw: Any):
        # whatever you need to do convert your model predictions back to dgp goes here
        # this should return dgp sample with required annotations as dgp annotation
        # objects
        return sample

    def autolabel_scene(self, scene_index, dataset, device='cuda:0'):

        # open the raw scene
        sample_idx = sorted([
            i for i, (scene_idx, _, _) in enumerate(dataset.dataset_item_index) if scene_idx == scene_index
        ])

        # We mostly expect models to be pytorch models, but we support other callable functions also
        if isinstance(self.model, nn.Module):
            self.model.to(device)
            self.model.eval()

        outputs_dgp = []

        prior_outputs = deque([], self.ar_steps)
        prior_samples = deque([], self.ar_steps)

        logging.info('running forward pass')
        for i in tqdm(sample_idx):

            # TODO: replace this with a more efficient multiworker loader
            context = dataset[i]
            assert len(context) == 1, 'only length 1 context windows are supported'

            # predict
            sample = context[0]
            with torch.no_grad():
                prior_samples.append(sample)
                batch_sample = self.preprocess_dgp_sample(prior_samples, device=device)
                predictions_raw = self.predict(batch_sample, prior_outputs)
                sample_pred = self.postprocess_to_dgp(sample, predictions_raw)

            prior_outputs.append(predictions_raw)
            outputs_dgp.append(sample_pred)

        # TODO: if model supports a backward inference pass i.e. a bidirectional lstm
        # run the sequence but in reverse

        # postprocess
        logging.info('running global post process')
        self.outputs_dgp = self.post_processor(outputs_dgp)

        # finally save the new scene
        scene = DGPAutolabelScene(
            dataset,
            scene_index,
            self.model_name,
            autolabel_keys=self.autolabel_keys,
            ontology_table=self.ontology_table
        )

        scene.save(sample_idx, self.outputs_dgp)


class DGPTestLidarCuboidAutoLabeler(DGPAutoLabeler):
    def __init__(self, model, post_processor=None, model_name=None, ar_steps=1):
        super().__init__(
            model,
            autolabel_keys=['bounding_box_3d'],
            post_processor=post_processor,
            model_name=model_name,
            ar_steps=ar_steps
        )

        self.ontology = None

    def predict(self, batch_sample, outputs):
        #print(batch_sample)
        boxes = deepcopy(batch_sample[-1]['bounding_box_3d'])
        for box in boxes:
            box.attributes['score'] = str(1.0)
        return boxes

    def preprocess_dgp_sample(self, samples: List[List[Dict[str, Any]]], device):
        return samples[-1]

    def postprocess_to_dgp(self, sample, predictions_raw: Any):
        sample[-1]['bounding_box_3d'] = predictions_raw
        return sample


def convert_lidar_box_to_dgp(box, score, label):
    loc = box.center.detach().cpu().numpy().flatten()
    dims = box.dims.detach().cpu().numpy().flatten()
    rot_y = box.yaw.detach().cpu().numpy()

    radians = (
        -rot_y - np.pi / 2  # <<< this is for second, not sure if this needs np.pi/2 for ouroboros evaluate
    )
    pose = Pose(tvec=loc, wxyz=Quaternion(axis=(0, 0, 1), radians=radians))
    bbox_3d = BoundingBox3D(pose, dims)
    bbox_3d._instance_id = 0
    bbox_3d._class_id = label.item()
    bbox_3d._attributes['score'] = str(score.item())
    return bbox_3d


class DGPLidarCuboidAutoLabeler(DGPAutoLabeler):
    def predict(self, batch_sample, outputs):
        outputs = self.model.forward_test(
            points=[[batch_sample.squeeze()]], img_metas=[[{
                'box_type_3d': LiDARInstance3DBoxes
            }]]
        )
        return outputs

    def preprocess_dgp_sample(self, samples: List[List[Dict[str, Any]]], device):

        # Grab the lidar datum from the most recent sample
        datum_dict = {datum['datum_name'].lower(): datum for datum in samples[-1]}
        lidar = datum_dict['lidar']

        # Convert and stack the points
        pc = torch.Tensor(lidar['point_cloud']).unsqueeze(0).to(device)
        ex = torch.Tensor(lidar['extra_channels']).unsqueeze(0).to(device)

        batch = torch.cat([pc, ex], dim=-1)

        # Sort points by height
        sort_idx = torch.argsort(-batch[..., 2]).squeeze()
        batch = batch[:, sort_idx]

        return batch

    def postprocess_to_dgp(self, sample, predictions_raw: Any):

        new_sample = deepcopy(sample)
        boxes = predictions_raw[0]['boxes_3d']
        scores = predictions_raw[0]['scores_3d']
        labels = predictions_raw[0]['labels_3d']
        # TODO: do something better than +1 to get the contigous id (why do these start at 1???)
        boxlist = [ convert_lidar_box_to_dgp(boxes[i], scores[i], labels[i] +1) for \
          i in range(len(boxes)) if labels[i]>=0 ]

        ontology = self.ontology_table['bounding_box_3d']
        new_sample[-1]['bounding_box_3d'] = BoundingBox3DAnnotationList(ontology, boxlist)

        return new_sample
