# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
import os
from copy import deepcopy
from functools import partial
from queue import Empty, Queue

import carla
import numpy as np
from matplotlib.cm import get_cmap
from PIL import Image

from dgp.annotations.bounding_box_3d_annotation import \
    BoundingBox3DAnnotationList
from dgp.annotations.depth_annotation import \
    DenseDepthAnnotation as BrokenDenseDepthAnnotation
# semseg ontology https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
from dgp.annotations.ontology import (BoundingBoxOntology, SemanticSegmentationOntology)
from dgp.annotations.semantic_segmentation_2d_annotation import \
    SemanticSegmentation2DAnnotation
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_3d import BoundingBox3D

dir_path = os.path.dirname(os.path.realpath(__file__))
CARLA_ONTOLOGY_FOLDER = 'ontology'
semseg_ontology = SemanticSegmentationOntology.load(
    os.path.join(dir_path, CARLA_ONTOLOGY_FOLDER, 'semseg-ontology.json')
)
cuboid_ontology = BoundingBoxOntology.load(os.path.join(dir_path, CARLA_ONTOLOGY_FOLDER, 'cuboid-ontology.json'))

# TODO:
# instance seg
# 2d boxes from instance seg
# optical flow 3d
# semantic lidar
# radar sensor
# add keypoints
# add human pose skeleton to walker cuboids
# weather metadata
# GNSS sensor
# IMU sensor
# add traffic lights and traffic light state
# add additional box metadata i.e, velocity, accel, visibility, staticness
# un-hardcode ontologies
# map info in dgp map format
# support for different camera models
# expose controls for agent


# This fixes the depth annotation save, remove this once autolabel pr is merged which includes fix
class DenseDepthAnnotation(BrokenDenseDepthAnnotation):
    def save(self, save_dir):
        pointcloud_path = os.path.join(save_dir, '{}.npz'.format(self.hexdigest))
        np.savez_compressed(pointcloud_path, data=self.depth)
        return pointcloud_path


def carla_box_to_cuboid(carla_bbox, tag_id, instance_id=None):
    """Convert carla bounding box to dgp bounding box"""
    class_id = cuboid_ontology.class_id_to_contiguous_id[tag_id]
    attributes = {'static': str(True)}
    transform_box = carla.Transform(carla_bbox.location, carla_bbox.rotation)

    # dgp sizes is passed as WLH or yxz
    pose = carla_pose_to_dgp_pose(transform_box)

    sizes = 2 * np.array([carla_bbox.extent.y, carla_bbox.extent.x, carla_bbox.extent.z])
    box = BoundingBox3D(pose=pose, sizes=sizes, class_id=class_id, instance_id=instance_id, attributes=attributes)
    return box


def get_initial_static_boxes(world):
    """Get static boxes. These are bounding boxes that exist when the world loads and do not move, things like parked cars etc"""
    static_boxes = []
    instance_id = 1000000
    for k in ['Vehicles']:  # Note: we can export vegetation, buildings, poles, and much more also
        instance_id += 10
        v = cuboid_ontology.name_to_id[k]
        boxes = world.get_level_bbs(getattr(carla.CityObjectLabel, k))
        converted_boxes = [carla_box_to_cuboid(box, v, instance_id=instance_id) for box in boxes]
        static_boxes.extend(converted_boxes)
    return static_boxes


def carla_actor_to_cuboid(actor):
    """Converts a carla actor to a dgp cuboid"""
    instance_id = actor.id
    class_tags = actor.semantic_tags
    if len(class_tags) == 0:
        tag = 3
    else:
        tag = class_tags[0]

    class_id = cuboid_ontology.class_id_to_contiguous_id[tag]
    attributes = {'type_id': actor.type_id}
    carla_bbox = actor.bounding_box
    transform_box = carla.Transform(carla_bbox.location, carla_bbox.rotation)
    transform_actor = actor.get_transform()

    # dgp sizes is passed as WLH or yxz
    pose_box = carla_pose_to_dgp_pose(transform_box)
    pose_actor = carla_pose_to_dgp_pose(transform_actor)

    pose = pose_actor * pose_box
    sizes = 2 * np.array([carla_bbox.extent.y, carla_bbox.extent.x, carla_bbox.extent.z])
    box = BoundingBox3D(pose=pose, sizes=sizes, class_id=class_id, instance_id=instance_id, attributes=attributes)
    return box


# TODO: refactor all of this
ANNOTAION_TO_SENSOR_TYPE = {
    'rgb': 'camera.rgb',
    'depth': 'camera.depth',
    'semantic_segmentation_2d': 'camera.semantic_segmentation',
    'point_cloud': 'point_cloud',
}

SENSOR_TYPE_TO_ANNOTATION = {v: k for k, v in ANNOTAION_TO_SENSOR_TYPE.items()}

# For the cameras we need to right apply this B matrix to get the axes swapped
# The problem is this is not a rotation (det = -1), so we have to break this out
# into a handedness flip and a rotation, and then apply the handedness to all points/vectors
# Note: we only have to apply this 'swap' to the camera poses
B = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float32)
Flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
Swap = np.linalg.inv(Flip) @ B


def estimate_camera_matrix(fov, h, w):
    """Build a camera matrix from fov and shape"""
    f = w / (2 * np.tan(.5 * fov * np.pi / 180))
    cx, cy = w / 2, h / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K


def carla_pose_to_dgp_pose(transform, swap_axes=False):
    """Convert a carla transform object to a dgp pose object"""
    M = np.array(transform.get_matrix())
    if swap_axes:
        M = M @ np.linalg.inv(Swap)

    M = Flip @ M @ np.linalg.inv(Flip)

    R, t = M[:3, :3], M[:3, -1]

    # Due to numerical issues, Pose.from_matrix errors out, so we have to manually re-orthogonalize our rotation matrix
    u, s, v = np.linalg.svd(R)
    R = u @ v
    p = Pose.from_rotation_translation(R, t)

    return p


def carla_rgb_to_img(sensor_data, use_pil=False):
    """Convert carla image to dgp image"""
    h, w = sensor_data.height, sensor_data.width
    # Note: carla images are in BGRA
    img = np.array(sensor_data.raw_data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3][:, :, ::-1]
    if use_pil:
        img = Image.fromarray(img)

    return {'rgb': img}


def carla_depth_to_depth(sensor_data):
    """Convert carla depth map to dgp depth"""
    h, w = sensor_data.height, sensor_data.width
    img = np.array(sensor_data.raw_data, dtype=np.float32).reshape(h, w, -1)[:, :, :3]  # this is in BGR

    # Note: this assumes the default carla clip plane is set to 1000
    normalized = (img[:, :, 2] + img[:, :, 1] * 256 + img[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return {'depth': DenseDepthAnnotation(in_meters.copy())}


def carla_semseg_to_semseg(sensor_data):
    """Convert carla semseg to dgp semseg"""
    h, w = sensor_data.height, sensor_data.width
    img = np.array(sensor_data.raw_data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
    # carla encodes label image in red channel but img is in BGR
    semseg = SemanticSegmentation2DAnnotation(semseg_ontology, img[:, :, 2].copy())
    return {'semantic_segmentation_2d': semseg}


def carla_metadata(sensor_data, swap_axes=False):
    """Get basic datum metdata for dgp"""
    pose = carla_pose_to_dgp_pose(sensor_data.transform, swap_axes=swap_axes)
    return {
        'timestamp': int(sensor_data.timestamp * 1e6),  # microseconds
        'frame': sensor_data.frame,
        'pose': pose
    }


def carla_lidar_to_lidar(sensor_data):
    """Convert carla lidar to dgp lidar"""
    points = np.frombuffer(sensor_data.raw_data, dtype=np.float32).reshape(-1, 4)
    point_cloud = (Flip[:3, :3] @ points[:, :3].copy().T).T
    reflectance = points[:, 3].copy().reshape(-1, 1)
    return {'point_cloud': point_cloud, 'extra_channels': reflectance}


def camera_callback(sensor_data, name, Q):
    Q.put((name, sensor_data))


def lidar_callback(sensor_data, name, Q):
    #print('*', end=',')
    Q.put((name, sensor_data))


def draw_depth_image(mtx: np.ndarray, p_wc: np.ndarray, Xw: np.ndarray, shape: list) -> np.ndarray:
    # p_wc: camera['pose'] Xw: points in world frame i.e. lidar['pose'] * lidar['point_cloud']
    # generate a depth image from a set of points
    P = mtx @ np.linalg.inv(p_wc)[:3, :]

    #project
    uvd = (P @ np.hstack((Xw, np.ones((len(Xw), 1), np.float32))).T).T
    z_c = np.expand_dims(uvd[:, 2], -1)
    uv = uvd[:, :2] / z_c
    uv = uv.astype(np.int32)

    H, W = shape[:2]
    depth = np.zeros((H, W, 1), dtype=np.float32)

    #filter for in view
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c[:, 0] > 0])
    uv, z_c = uv[in_view], z_c[in_view]
    depth[uv[:, 1], uv[:, 0]] = z_c

    return depth, uv


class CarlaDGPAgent():
    def __init__(self, world, model, color, transform, datums, static_boxes=None):
        """Carla agent that outputs oberservations in SynchronizedScene format

        Parameters
        ----------
        world: carla world object

        model: str
            Carla blueprint to use for main ego model
        
        color: np.ndarray
            RGB to use for agent color
        
        transform: carla transform
            Intial spawn pose for agent in world coordinates
        
        datums: List[dict]
            List of datums, parameters and annotations see configs/car-dev.json for example
        
        static_boxes: List
            List of boxes available when world was first loaded
        """
        self.world = world
        self.actor_list = []
        self.Q = Queue()

        vehicle = self.get_main_actor(model, transform, color=color)
        self.actor_list.append(vehicle)

        self.intrinsics_lookup = {}
        self.calibration_table = {}
        self.ontology_table = {'semantic_segmentation_2d': semseg_ontology, 'bounding_box_3d': cuboid_ontology}

        self.static_boxes = static_boxes

        self.synthetic_lidar_extrinsics = {}

        for datum in datums:
            datum_type = datum.pop('datum_type')
            if datum_type == 'image':
                camera_actors, intrinsics, extrinsics = self.get_camera_sensor(**datum)
                self.actor_list.extend(camera_actors)
                self.intrinsics_lookup[datum['name']] = intrinsics
                self.calibration_table[datum['name']] = carla_pose_to_dgp_pose(extrinsics, swap_axes=True)

            elif datum_type == 'point_cloud':
                lidar, extrinsics = self.get_lidar_sensor(**datum)
                self.actor_list.append(lidar)
                ext = carla_pose_to_dgp_pose(extrinsics)
                self.synthetic_lidar_extrinsics[datum['name']] = ext
                self.calibration_table[datum['name']] = ext.inverse() * ext

    @property
    def ego(self):
        return self.actor_list[0]

    def get_main_actor(self, model: str, transform, color=None):
        """Returns the main model for this agent"""
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find(model)
        if bp.has_attribute('color'):
            color_options = bp.get_attribute('color').recommended_values
            if color is None:
                color = np.random.choice(color_options)

            bp.set_attribute('color', color)

        vehicle = self.world.spawn_actor(bp, transform)
        return vehicle

    def get_lidar_sensor(self, name, tvec, rvec):

        tx, ty, tz = tvec
        roll, pitch, yaw = rvec

        transform = carla.Transform(carla.Location(x=tx, y=ty, z=tz), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

        bp_library = self.world.get_blueprint_library()
        # TODO: move these into the car spec json
        bp = bp_library.find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(200.0))
        bp.set_attribute('channels', str(128))
        #bp.set_attribute('horizontal_fov', str(90))
        bp.set_attribute('upper_fov', str(20.0))
        bp.set_attribute('lower_fov', str(-25.0))
        bp.set_attribute('rotation_frequency', str(10))
        bp.set_attribute('points_per_second', str(200000))
        bp.set_attribute('sensor_tick', str(.1))

        actor = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        sensor_type = ANNOTAION_TO_SENSOR_TYPE['point_cloud']
        sensor_name = f'{name}.{sensor_type}'

        f = partial(lidar_callback, name=sensor_name, Q=self.Q)

        actor.listen(f)
        return actor, transform

    def get_camera_sensor(self, name, height, width, fov, tvec, rvec, annotations=None, attributes=None):
        tx, ty, tz = tvec
        roll, pitch, yaw = rvec

        transform = carla.Transform(carla.Location(x=tx, y=ty, z=tz), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

        intrinsics = estimate_camera_matrix(fov, height, width)
        # TODO: add distortion params if set in attributes

        if annotations is None:
            annotations = []

        if attributes is None:
            attributes = {}

        if 'rgb' not in annotations:
            annotations.append('rgb')

        bp_library = self.world.get_blueprint_library()

        actor_list = []
        for annotation in annotations:
            sensor_type = ANNOTAION_TO_SENSOR_TYPE[annotation]
            bp = bp_library.find(f'sensor.{sensor_type}')
            bp.set_attribute('image_size_y', str(height))
            bp.set_attribute('image_size_x', str(width))
            bp.set_attribute('fov', str(fov))
            bp.set_attribute('sensor_tick', str(.1))

            # add other properties
            for attribute, attribute_value in attributes.items():
                bp.set_attribute(attribute, str(attribute_value))

            actor = self.world.spawn_actor(bp, transform, attach_to=self.ego)

            sensor_name = f'{name}.{sensor_type}'
            f = partial(camera_callback, name=sensor_name, Q=self.Q)

            actor.listen(f)
            actor_list.append(actor)

        return actor_list, intrinsics, transform

    def get_sample(self, frame):
        """Get a sample for a specific frame"""
        sample_dict = {}
        found = 0
        counter = 0
        max_it = 10000
        while (found < len(self.actor_list) - 1) and counter < max_it:
            counter += 1
            try:
                name, sensor_data = self.Q.get(True, 20.0)
            except Empty:
                #print('empty Q')
                continue

            if sensor_data.frame != frame:
                continue

            base_name = name.split('.')[0]
            if base_name not in sample_dict:
                sample_dict[base_name] = {}

            sample_dict[base_name][name.replace(base_name + '.', '')] = sensor_data
            found += 1

        if found != len(self.actor_list) - 1:
            print('missing readings', found, len(self.actor_list) - 1, counter, self.Q.qsize())
            return None

        return self._format_sample(sample_dict)

    def _format_sample(self, sample_dict):
        datums = sorted(sample_dict.keys())
        sample = []
        for datum in datums:
            sensors = sorted(sample_dict[datum].keys())

            results = {'datum_name': datum}

            for sensor in sensors:
                sensor_data = sample_dict[datum][sensor]
                annotation_type = SENSOR_TYPE_TO_ANNOTATION[sensor]

                if annotation_type == 'rgb':
                    results.update(carla_metadata(sensor_data, swap_axes=True))
                    results.update(carla_rgb_to_img(sensor_data, use_pil=True))
                    results.update({'datum_type': 'image'})
                    results.update({
                        'intrinsics': self.intrinsics_lookup[datum],
                        'extrinsics': self.calibration_table[datum]
                    })

                elif annotation_type == 'depth':
                    results.update(carla_depth_to_depth(sensor_data))
                elif annotation_type == 'semantic_segmentation_2d':
                    results.update(carla_semseg_to_semseg(sensor_data))
                elif annotation_type == 'point_cloud':
                    results.update(carla_metadata(sensor_data))
                    results.update(carla_lidar_to_lidar(sensor_data))
                    ext = self.synthetic_lidar_extrinsics[datum]
                    # Move things into the virtual lidar frame (ego local)
                    # TODO: expand this for multiple lidars
                    results['pose'] = ext.inverse() * results['pose']
                    results['point_cloud'] = ext * results['point_cloud']
                    results.update({'datum_type': 'point_cloud'})
                    results.update({'extrinsics': self.calibration_table[datum]})

                    # TODO: make sure these get_actors return information at the same instant
                    # as the lidar, also refactor this whole thing because this is really messy
                    vehicles = self.world.get_actors().filter('vehicle.*')
                    walkers = self.world.get_actors().filter('walker.*')
                    ego_id = self.ego.id
                    #print('getting cuboids...')

                    max_r = 150
                    # TODO: add visibilty check using either raycasted intersection with cuboid, point in cuboid
                    # or instance mask in camera
                    boxes = []
                    for actors in [vehicles, walkers]:
                        for actor in actors:
                            if actor.id == ego_id:
                                continue
                            box = carla_actor_to_cuboid(actor)

                            box._pose = results['pose'].inverse() * box.pose
                            R = np.linalg.norm(box.pose.tvec)
                            if R < max_r:
                                boxes.append(box)

                    if self.static_boxes is not None:
                        sboxes = [
                            deepcopy(box)
                            for box in self.static_boxes
                            if np.linalg.norm(results['pose'].tvec - box.pose.tvec) < max_r
                        ]
                        for box in sboxes:
                            box._pose = results['pose'].inverse() * box.pose
                        boxes.extend(sboxes)

                    if len(boxes) > 0:
                        boxann = BoundingBox3DAnnotationList(cuboid_ontology, boxes)
                        results['bounding_box_3d'] = boxann

            sample.append(results)

        return sample

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
