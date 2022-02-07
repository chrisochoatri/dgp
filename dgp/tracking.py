from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dgp.annotations.bounding_box_3d_annotation import (BoundingBox3D, BoundingBox3DAnnotationList)
from dgp.utils.pose import Pose


def vel_to_rot(vel):
    """Convert velocity to rotation matrix"""
    # TODO: double check. Also this fails if velocity is 0
    up = np.array([0, 0.0, 1.0])
    u = vel / np.linalg.norm(vel)
    v = -1 * np.cross(u, up)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return np.stack([u, v, w], -1)


class KalmanFilterTrack():
    def __init__(
        self,
        process_noise_args: Dict[str, Any],
        measurement_noise_args: Dict[str, Any],
        initial_sigma_args: Dict[str, Any],
    ):
        """Base class for dgp BoundingBox3D Kalman filters"""
        self.observations = []
        self.mus = []
        self.sigmas = []
        self.Qs = []
        self.Fs = []
        self.sample_idx = []

        self.hits = 0
        self.misses = 0
        self.status = 'new'
        self.score = None
        self.class_id = None
        self.process_noise_args = process_noise_args
        self.measurement_noise_args = measurement_noise_args
        self.initial_sigma_args = initial_sigma_args
        self.mu = None
        self.sigma = None

    def initial_mu(self, box: BoundingBox3D) -> np.ndarray:
        """Inits state"""
        raise NotImplementedError

    def initial_sigma(self, box: BoundingBox3D) -> np.ndarray:
        """Inits covariance"""
        raise NotImplementedError

    def measurement_model(self) -> np.ndarray:
        """Returns matrix to project state to measurement (H)"""
        raise NotImplementedError

    def measurement_noise(self, measurement_noise_args: Dict[str, Any]) -> np.ndarray:
        """Returns measurement covariance matrix"""
        raise NotImplementedError

    def process_model(self, mu: np.ndarray, dt: float) -> np.ndarray:
        """Returns state transition matrix F"""
        raise NotImplementedError

    def process_noise(self, mu: np.ndarray, dt: float, process_noise_args: Dict[str, Any]) -> np.ndarray:
        """Returns process noise covariance Q"""
        raise NotImplementedError

    def box_to_state(self, box: BoundingBox3D) -> np.ndarray:
        """Converts a bounding box 3d object into a state (mean) vector"""
        raise NotImplementedError

    def state_to_box(self, mu: np.ndarray, sigma: np.ndarray) -> BoundingBox3D:
        """Converts a state vector to a bounding box 3d"""
        raise NotImplementedError

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts new cuboid state using process model F and noise Q. Returns belief center and covariance"""
        Q = self.process_noise(self.mu, dt, self.process_noise_args)
        F = self.process_model(self.mu, dt)
        mu = F @ self.mu
        sigma = F @ self.sigma @ F.T + Q

        self.Qs.append(Q)
        self.Fs.append(F)
        self.mu = mu
        self.sigma = sigma
        return mu, sigma

    def update(self, box: Optional[BoundingBox3D], sample_idx, init=False):
        """Updates cuboid state with new observations. Tracks hits/misses"""
        if box is not None:
            self.misses = 0
            self.hits += 1

            if init == False:
                z = self.box_to_state(box)
                H = self.measurement_model()
                R = self.measurement_noise(self.measurement_noise_args)

                y = z - H @ self.mu
                K = (self.sigma @ H.T) @ np.linalg.inv(H @ self.sigma @ H.T + R)
                mu = self.mu + K @ y

                n = self.sigma.shape[0]
                sigma = (np.eye(n) - K @ H) @ self.sigma
                #print(sigma[3,3])
            else:
                mu = self.initial_mu(box)
                sigma = self.initial_sigma(box)

            if self.score is None:
                self.score = float(box.attributes.get('score', 1.0))
            else:
                self.score = np.sqrt(self.score * float(box.attributes.get('score', 1.0)))

            if self.class_id is None:
                self.class_id = box.class_id

        else:
            mu, sigma = self.mu, self.sigma
            self.misses += 1
            self.hits = 0

        self.observations.append(box)
        self.mu = mu
        self.sigma = sigma
        self.mus.append(mu)
        self.sigmas.append(sigma)
        self.sample_idx.append(sample_idx)

        return mu, sigma

    def mahalnobis(self, boxes: Union[List[BoundingBox3D], BoundingBox3DAnnotationList]) -> np.ndarray:
        """Return distance between box observations and most recent uncertain track state"""
        H = self.measurement_model()
        R = self.measurement_noise(self.measurement_noise_args)
        VI = np.linalg.inv(H @ self.sigma @ H.T)
        mu = H @ self.mu
        observations = [self.box_to_state(box) for box in boxes]
        return np.array([np.sqrt((z - mu) @ VI @ (z - mu).T) for z in observations])

    def get_state_by_sample_idx(self, sample_idx: int):
        """Helper function to associate track state to global (external) sample index.
        Required to map tracks back to dgp samples."""
        # get max good observation
        valid_idx = np.array([i for i, box in enumerate(self.observations) if box is not None])

        try:
            idx = self.sample_idx.index(sample_idx)
        except:
            idx = -1

        if idx > np.max(valid_idx):
            # we have state for this sample but it is past our last observation
            return None

        if idx >= 0:
            box = self.observations[idx]
            if box is None:
                # get the nearest box
                deltas = np.abs(idx - valid_idx)
                new_idx = valid_idx[np.argmin(deltas)]
                box = self.observations[new_idx]
            return self.mus[idx], self.sigmas[idx], box

    def smooth(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run filter backward with fixed association to smooth states. Implements Rauch-Tung-Striebel smoothing."""
        # Adapted from https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/13-Smoothing.ipynb
        N = len(self.mus)
        m = self.mu.shape[0]

        K = np.zeros((N, m, m))
        x, P = deepcopy(self.mus), deepcopy(self.sigmas)

        for k in range(N - 2, -1, -1):
            # predict
            Pp = self.Fs[k] @ P[k] @ self.Fs[k].T + self.Qs[k]

            # update
            K[k] = P[k] @ self.Fs[k].T @ np.linalg.pinv(Pp)
            x[k] += K[k] @ (x[k + 1] - (self.Fs[k] @ x[k]))
            P[k] += K[k] @ (P[k + 1] - Pp) @ K[k].T

        return x, P


class ConstantVelocityKF(KalmanFilterTrack):
    # State space is x,y,z, vx,vy,vz, l,h,w
    def measurement_model(self):
        # only measure x,y,z, l, h,w   (6,9) ( 9,1) = (6,1)
        # yapf: disable
        H= np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        # yapf: enable
        return H

    def measurement_noise(self, measurement_noise_args):
        # TODO: make the noise depend on distace from ego
        sx = measurement_noise_args['measurement_noise_position_var']
        sd = measurement_noise_args['measurement_noise_dim_var']
        R = np.eye(6)
        R[:3, :3] *= sx
        R[3:, 3:] *= sd
        return R

    def process_model(self, mu, dt):
        # yapf: disable
        F = np.array([
            [1, 0, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        # yapf: enable
        return F

    def process_noise(self, mu, dt, process_noise_args):
        sv = process_noise_args['process_noise_velocity_var']
        sd = process_noise_args['process_noise_dim_var']
        # yapf: disable
        Qv = sv * np.array([
            [.25 * dt**4,0              ,0              ,.5 * dt**3     ,0          ,0          ],
            [0          ,.25 * dt**4    ,0              ,0              ,.5 * dt**3 ,0          ],
            [0          ,0              ,.25 * dt**4    ,0              ,0          ,.5 * dt**3 ],
            [.5 * dt**2 ,0              ,0              ,dt**2          ,0          ,0          ],
            [0          ,.5 * dt**2     ,0              ,0              ,dt**2      ,0          ],
            [0          ,0              ,.5 * dt**2     ,0              ,0          ,dt**2      ],
        ])
        # yapf: enable

        Q = np.zeros((9, 9))
        Q[:6, :6] = Qv
        Q[6:, 6:] = sd * np.eye(3, 3)
        return Q

    def box_to_state(self, box: BoundingBox3D) -> np.ndarray:
        return np.concatenate([box.pose.tvec, box.sizes])

    def state_to_box(self, mu, sigma, org_box, instance_id):
        tvec = mu[:3]
        vel = mu[3:6]
        dims = mu[6:9]
        R = vel_to_rot(vel)
        speed = np.linalg.norm(vel)  # TODO: maybe just x,y plane speed?
        if speed < 1 and org_box is not None:
            R = org_box.pose.rotation_matrix

        pose = Pose.from_rotation_translation(R, tvec)
        #print('tvec', pose.tvec, 'vel', vel)
        box = BoundingBox3D(pose, dims, self.class_id, instance_id=instance_id)
        box.attributes['score'] = str(self.score)
        box.attributes['speed'] = str(speed)
        return box

    def initial_mu(self, box):
        heading = box.pose.rotation_matrix[:, 0]
        heading[2] = 0
        return np.concatenate([box.pose.tvec, heading, box.sizes])

    def initial_sigma(self, box):
        sigma_x, sigma_v, sigma_d = self.initial_sigma_args['sigma_x'], self.initial_sigma_args[
            'sigma_v'], self.initial_sigma_args['sigma_d']
        sig = np.concatenate([sigma_x, sigma_v, sigma_d])
        sigma = np.eye(len(sig))
        np.fill_diagonal(sigma, sig)
        return sigma


class Tracker():
    last_track_id = 0

    def __init__(
        self,
        track_type,
        process_noise_args,
        measurement_noise_args,
        initial_sigma_args,
        min_hits_to_track=5,
        max_misses_to_forget_new=2,
        max_misses_to_forget_tracked=10,
        match_threshold=30,
        min_box_score=.1,
        cuboid_key='bounding_box_3d',
        cuboid_datum='lidar',
    ):
        self.min_hits_to_track = min_hits_to_track
        self.max_misses_to_forget_new = max_misses_to_forget_new
        self.max_misses_to_forget_tracked = max_misses_to_forget_tracked
        self.match_threshold = match_threshold
        self.min_box_score = min_box_score
        self._tracks = dict()
        self.valid_track_ids = []
        self.prior_t = None
        self.track_type = track_type
        self.process_noise_args = process_noise_args
        self.measurement_noise_args = measurement_noise_args
        self.initial_sigma_args = initial_sigma_args
        self.cuboid_key = cuboid_key
        self.cuboid_datum = cuboid_datum

    def register_track(self, box, sample_index):
        track = self.track_type(self.process_noise_args, self.measurement_noise_args, self.initial_sigma_args)
        track.update(box, sample_index, init=True)
        Tracker.last_track_id += 1  # dont start at 0, we reserve that for ego
        self._tracks[Tracker.last_track_id] = track

    def update_track_status(self):
        # Go through each track and apply the thresholds for continous hits/misses
        junk = []
        for k, v in self._tracks.items():
            if v.status == 'new':
                if v.hits >= self.min_hits_to_track:
                    v.status = 'tracking'
                    self.valid_track_ids.append(k)
                    #print('tracking', k, '!!!')

                elif v.misses >= self.max_misses_to_forget_new:
                    v.status = 'junk'
                    junk.append(k)
                    #print('junk', k)

            elif v.status == 'tracking':
                if v.misses >= self.max_misses_to_forget_tracked:
                    v.status = 'lost'
                    #print('lost',k)

            elif v.status == 'lost':
                # TODO remove ancient tracks
                pass

        # Remove false tracks
        for k in junk:
            del self._tracks[k]

    def get_current_track_ids(self):
        return [k for k, v in self._tracks.items() if v.status in ('new', 'tracking')]

    def match(self, D, thresh=10):

        matches = linear_sum_assignment(D)

        # try:
        #     Dvalid = [D[i,j] for i,j in zip(*matches)]
        #     print('input', np.mean(D), np.max(D), np.min(D))
        #     print('raw', np.mean(Dvalid), np.max(Dvalid), np.min(Dvalid))
        # except:
        #     pass

        valid = []
        for row, col in zip(*matches):
            if D[row, col] <= thresh:
                valid.append((row, col))
        return valid

    def step(self, new_observations, lidar_pose, sample_index, t):
        # convert all new_observations to global static frame
        for box in new_observations:
            box._pose = lidar_pose * box.pose

        # Get the candidate tracks (new and tracking)
        current_track_ids = self.get_current_track_ids()

        if self.prior_t is None:
            self.prior_t = t

        # Predict their new positions
        for k in current_track_ids:
            self._tracks[k].predict(t - self.prior_t)

        # match up current tracked items to new observations
        N = len(current_track_ids)
        M = len(new_observations)
        D = np.zeros((N, M))
        for i, k in enumerate(current_track_ids):
            D[i, :] = self._tracks[k].mahalnobis(new_observations)

        # classes_current = [self._tracks[k].class_id for k in current_track_ids]
        # classes_new = [box.class_id for box in new_observations]

        # Dclass = 10*cdist( np.array(classes_current).reshape(-1,1) , np.array(classes_new).reshape(-1,1), metric='hamming') # 0 same 1 different
        # D += Dclass

        matches = self.match(D, self.match_threshold)

        # # filter again for same class
        # matches = [(i,j) for i, j in matches if Dclass[i,j] ==0]

        for match_old_idx, match_new_idx in matches:
            k = current_track_ids[match_old_idx]
            new_observation = new_observations[match_new_idx]
            self._tracks[k].update(new_observation, sample_index)

        # Matches: add observations
        # if matches is length 0 then nothing was matched
        if len(matches) == 0:
            match_old, match_new = [], []
        else:

            Dvalid = [D[i, j] for i, j in matches]
            #print(np.mean(Dvalid), np.max(Dvalid), np.min(Dvalid))

            match_old, match_new = zip(*matches)

        # Not matched current: add None observation
        unmatched_old = [k for i, k in enumerate(current_track_ids) if i not in match_old]
        for k in unmatched_old:
            self._tracks[k].update(None, sample_index)

        # Not matched new_observations: register
        unmatched_new = [i for i, obs in enumerate(new_observations) if i not in match_new]
        for i in unmatched_new:
            obs = new_observations[i]
            #print('heading', heading)
            self.register_track(obs, sample_index=sample_index)

        self.prior_t = t
        self.update_track_status()

    def run(self, samples):
        # samples: list[ dgp samples]
        # lidar -1 for now

        # run forward
        for sample_index, sample in enumerate(samples):
            datum_dict = {datum['datum_name'].lower(): datum for datum in sample}
            lidar = datum_dict[self.cuboid_datum]
            lidar_pose = lidar['pose']
            t = lidar['timestamp'] / 1e7
            observations = lidar[self.cuboid_key].boxlist
            self.step(observations, lidar_pose, sample_index, t)

        # smooth
        for k in self.valid_track_ids:
            self._tracks[k].smooth()

        # final filter
        self.valid_track_ids = [k for k in self.valid_track_ids if self._tracks[k].score >= self.min_box_score]

        # convert back to BoundingBox3D
        for sample_index, sample in enumerate(samples):
            lidar = sample[-1]
            lidar_pose_inv = lidar['pose'].inverse()
            #print('sample_index', sample_index)

            boxlist = []
            for k in self.valid_track_ids:
                v = self._tracks[k]
                # if v.score < self.min_box_score:
                #     #print('skipping low score', v.score)
                #     continue

                state = v.get_state_by_sample_idx(sample_index)
                if state is not None:
                    mu, sigma, box = state
                    box = v.state_to_box(mu, sigma, box, k)
                    box._pose = lidar_pose_inv * box.pose
                    boxlist.append(box)

            sample[-1][self.cuboid_key].boxlist = boxlist

        return samples

    def __call__(self, samples):
        new_samples = self.run(samples)
        print(f'generated {len(self.valid_track_ids)} tracks')
        return new_samples
