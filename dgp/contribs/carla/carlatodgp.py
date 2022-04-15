# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
import hashlib
import logging
import os
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List

import cv2
import numpy as np

from dgp import (
    BOUNDING_BOX_2D_FOLDER, BOUNDING_BOX_3D_FOLDER, CALIBRATION_FOLDER, DEPTH_FOLDER, INSTANCE_SEGMENTATION_2D_FOLDER,
    INSTANCE_SEGMENTATION_3D_FOLDER, ONTOLOGY_FOLDER, POINT_CLOUD_FOLDER, RADAR_POINT_CLOUD_FOLDER, RGB_FOLDER,
    SEMANTIC_SEGMENTATION_2D_FOLDER, SEMANTIC_SEGMENTATION_3D_FOLDER
)
from dgp.annotations import (ANNOTATION_REGISTRY,Annotation)
from dgp.constants import ANNOTATION_KEY_TO_TYPE_ID
from dgp.proto import geometry_pb2, point_cloud_pb2, radar_point_cloud_pb2
from dgp.proto.sample_pb2 import Datum, SampleCalibration
from dgp.proto.scene_pb2 import Scene
from dgp.utils.camera import pbobject_from_camera_matrix, pbobject_from_camera_matrix_and_distortion
from dgp.utils.pose import Pose
from dgp.utils.protobuf import (generate_uid_from_pbobject,save_pbobject_as_json)

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


@lru_cache(maxsize=None)
def _mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)


def _write_point_cloud(filename, X):
    """Utility function for writing point clouds."""
    _mkdir(os.path.dirname(filename))
    np.savez_compressed(filename, data=X.astype(np.float32))


def _write_image(filename, X):
    """Utility function for writing images."""
    _mkdir(os.path.dirname(filename))
    cv2.imwrite(filename, cv2.cvtColor(X, cv2.COLOR_RGB2BGR))


def _write_radar_point_cloud(filename, X):
    """Utility function for writing radar point clouds."""
    _mkdir(os.path.dirname(filename))
    np.savez_compressed(filename, data=X.astype(np.float32))


class DGPSceneConstructor():
    def __init__(self, output_directory):
        self.output_dir = output_directory
        self.scene = Scene()

        # Current sample we will be appending Datum's to until a new one is created
        self.current_sample = None
        self.calibration_for_current_sample = None

        # `prev_datum` is the previous datum which does NOT have a `next_key` populated.
        # Once `next_key` is populated the Datum gets dumped
        # Keys are ["LIDAR", "camera_01", ...]
        self.prev_datum = defaultdict(lambda: None)
        self.ontology_table = dict()

    def _save_calibration_table(self):
        """
        Save calibration table for current sample into `calibration` folder.
        """
        assert self.calibration_for_current_sample is not None

        # NOTE: this should not be changing from sample to sample, so all save's should collide
        # to same calibration file
        self.current_sample.calibration_key = generate_uid_from_pbobject(self.calibration_for_current_sample)

        calibration_output_filename = os.path.join(
            self.output_dir, CALIBRATION_FOLDER, "{}.json".format(self.current_sample.calibration_key)
        )

        # Make directory as this is the first time we are writing there and it doesn't exist
        _mkdir(os.path.dirname(calibration_output_filename))

        save_pbobject_as_json(self.calibration_for_current_sample, calibration_output_filename)

    def create_new_sample(self):
        """
        Create a new sample to add Datum's to.
        """

        # If a sample already exists then add a calibration key to it and dump the calibration file
        # to "calibration/<sha1>.json"
        if self.current_sample is not None:
            self._save_calibration_table()

        # Create new set of sample + calibration
        self.current_sample = self.scene.samples.add()
        self.calibration_for_current_sample = SampleCalibration()

    def save_ontology(self, ontology, annotation_type):
        """Saves the ontology files. If no ontology_table was passed, attemps to copy the required ontologies
        from the base scene"""
        ontology_dir = os.path.join(
            self.output_dir,
            ONTOLOGY_FOLDER,
        )
        os.makedirs(ontology_dir, exist_ok=True)

        if annotation_type not in self.ontology_table:
            self.ontology_table[annotation_type] = ontology
            ontology_type_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_type]
            ontology_path = self.ontology_table[annotation_type].save(ontology_dir)
            self.scene.ontologies[ontology_type_id] = os.path.basename(ontology_path).replace('.json', '')
            print(
                'saved ontology', ontology_type_id,
                os.path.basename(ontology_path).replace('.json', ''), ontology_dir
            )

    def save_annotation(self, datum, annotation: Annotation, annotation_key: str, datum_name: str) -> None:
        """Save a dgp annotation object"""

        assert isinstance(annotation, ANNOTATION_REGISTRY[annotation_key])

        annotation_folder = ANNOTATION_TYPE_ID_TO_FOLDER.get(annotation_key, annotation_key)
        save_dir = os.path.join(
            self.output_dir, annotation_folder, datum_name
        )  # TODO: this needs to be annotation_folder/datum_name
        os.makedirs(save_dir, exist_ok=True)

        datum_type = datum.WhichOneof('datum_oneof')
        datum_value = getattr(datum, datum_type)  # This is datum.image or datum.point_cloud etc

        annotation_type_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_key]

        annotation_path = annotation.save(save_dir)
        datum_value.annotations[annotation_type_id] = os.path.join(
            annotation_folder, datum_name, os.path.basename(annotation_path)
        )

    def add_camera_datum(self, datum):
        """
        Add data for a specific camera sensor
        """
        assert self.scene.name, "Need to add Point Cloud datum before Camera Datum's to correctly initialize scene name"

        camera_name = datum['datum_name']
        camera_datum = Datum()

        timestamp = int(datum['timestamp'])

        camera_datum.id.timestamp.FromMicroseconds(timestamp)
        camera_datum.id.log = 'carla'
        camera_datum.id.name = camera_name
        image_array = np.array(datum['rgb']).astype(np.uint8)

        camera_datum.key = hashlib.sha1(camera_datum.id.SerializeToString()).hexdigest()
        image = camera_datum.datum.image
        image.filename = os.path.join(RGB_FOLDER, camera_datum.id.name, "{}.jpeg".format(timestamp))
        image_output_filename = os.path.join(self.output_dir, image.filename)
        _write_image(image_output_filename, image_array)
        image.height, image.width, image.channels = image_array.shape

        # Add `next_key` to previous Datum of this camera
        prev_datum = self.prev_datum[camera_name]
        if prev_datum is not None:
            prev_datum.next_key = camera_datum.key
            self.scene.data.extend([prev_datum])
            camera_datum.prev_key = prev_datum.key

        # Add local pose at this camera's timestamp to Datum
        pose_LS = datum['pose'].to_proto()
        image.pose.CopyFrom(pose_LS)

        # Camera intrinsics
        self.calibration_for_current_sample.names.extend([camera_datum.id.name])
        intrinsics = datum['intrinsics']
        distortion = datum.get('distortion', None)
        self.calibration_for_current_sample.intrinsics.extend([pbobject_from_camera_matrix_and_distortion(intrinsics, distortion)])

        # Camera extrinsics
        extrinsics_VS = datum['extrinsics'].to_proto()
        self.calibration_for_current_sample.extrinsics.extend([extrinsics_VS])

        # annotations
        if 'depth' in datum:
            self.save_annotation(camera_datum.datum, datum['depth'], annotation_key='depth', datum_name=camera_name)

        if 'semantic_segmentation_2d' in datum:
            self.save_annotation(
                camera_datum.datum,
                datum['semantic_segmentation_2d'],
                annotation_key='semantic_segmentation_2d',
                datum_name=camera_name
            )

        # Store current Datum so that its `next_key` is populated in next call to ``add_camera_datum``
        self.prev_datum[camera_name] = camera_datum

        # Add Datum key to sample
        self.current_sample.datum_keys.extend([camera_datum.key])

    def add_point_cloud_datum(self, datum):
        # Name the scene on first call to update, "timestamp" is first point cloud
        # timestamp in scene. TODO (allan.raventos): `self.scene.description`
        timestamp = int(datum['timestamp'])

        if not self.scene.name:
            self.scene.name = "{}_{}".format('carla', timestamp)
            self.scene_name = self.scene.name.replace("/", "-")
            self.scene.log = 'carla'

        # NOTE: not adding `current_sample.id.index` (absolute timestamp for each sensor
        # is sufficient)
        self.current_sample.id.timestamp.FromMicroseconds(timestamp)
        self.current_sample.id.log = 'carla'

        sensor_name = datum['datum_name']
        point_cloud = np.hstack([datum['point_cloud'], datum['extra_channels'].reshape(-1, 1)])
        pc_datum = Datum()

        # Datum ID
        pc_datum.id.CopyFrom(self.current_sample.id)
        pc_datum.id.name = sensor_name

        pc_datum.key = hashlib.sha1(pc_datum.id.SerializeToString()).hexdigest()
        pc = pc_datum.datum.point_cloud
        pc.filename = os.path.join(POINT_CLOUD_FOLDER, pc_datum.id.name, "{}.npz".format(timestamp))

        pc_output_filename = os.path.join(self.output_dir, pc.filename)
        _write_point_cloud(pc_output_filename, point_cloud)
        pc.point_format.extend([
            point_cloud_pb2.PointCloud.X, point_cloud_pb2.PointCloud.Y, point_cloud_pb2.PointCloud.Z,
            point_cloud_pb2.PointCloud.INTENSITY
        ])

        # Point cloud pose at point cloud timestamp (representative timestamp)
        pose_LS = datum['pose'].to_proto()
        pc.pose.CopyFrom(pose_LS)

        # Add `next_key` to previous point cloud Datum
        prev_datum = self.prev_datum[sensor_name]
        if prev_datum is not None:
            prev_datum.next_key = pc_datum.key
            self.scene.data.extend([prev_datum])
            pc_datum.prev_key = prev_datum.key

        # Store current Datum so that its `next_key` is populated in next call to ``add_point_cloud_datum``
        self.prev_datum[sensor_name] = pc_datum

        # Add datum key to sample
        self.current_sample.datum_keys.extend([pc_datum.key])

        # Point cloud intrinsics (empty)
        self.calibration_for_current_sample.names.extend([pc_datum.id.name])
        self.calibration_for_current_sample.intrinsics.extend([geometry_pb2.CameraIntrinsics()])

        # Point cloud extrinsics
        extrinsics_VS = datum['extrinsics'].to_proto()
        self.calibration_for_current_sample.extrinsics.extend([extrinsics_VS])

        if 'bounding_box_3d' in datum:
            #print('saving cuboids...')
            self.save_annotation(
                pc_datum.datum, datum['bounding_box_3d'], annotation_key='bounding_box_3d', datum_name=sensor_name
            )

    def dump_scene(self):
        """Dump accumulated Scene after adding to it any Datum's still in `self.prev_datum`
        """

        # There's a chance that we have a calibration table that has not been saved
        if self.current_sample is not None:
            self._save_calibration_table()

        # Add last Datum of each sensor to data
        for prev_datum in self.prev_datum.values():
            assert not prev_datum.next_key
            self.scene.data.extend([prev_datum])

        # Print number of datums for each sensor type. Camera datums might not available early in the TLog
        # and might also be dropped somewhere in the middle of the TLog.
        datum_name_to_count = Counter([_datum.id.name for _datum in self.scene.data])
        print('-' * 80)
        print('From `DGPSceneConstructor.dump_scene`:')
        for datum_name, count in datum_name_to_count.items():
            print('Datum name "{}" has {} datums'.format(datum_name, count))
        print('-' * 80)

        assert len(set(datum_name_to_count.values())) == 1, 'all samples should have the same number of datums'

        # Check for duplicate datum ids
        datum_key_to_count = Counter([_datum.key for _datum in self.scene.data])
        dupe_count = 0
        for datum_name, count in datum_key_to_count.items():
            if count > 1:
                print('Datum key "{}" has {} datums'.format(datum_name, count))
                dupe_count += 1
        print('-' * 80)

        assert dupe_count == 0, "duplicate datum keys were found"

        # Compute a scene hash
        scene_hash = generate_uid_from_pbobject(self.scene)

        # Finally dump scene
        scene_output_filename = os.path.join(self.output_dir, "scene_{}.json".format(scene_hash))
        save_pbobject_as_json(self.scene, scene_output_filename)
