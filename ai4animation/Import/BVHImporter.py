# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re

import numpy as np
from ai4animation.Animation.Motion import Hierarchy, Motion
from ai4animation.Math import Rotation, Tensor, Transform

channelmap = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}

def _euler_to_rotation_matrix(angles, order):
    angles = Tensor.Create(angles)

    axis_to_rotation = {
        "x": Rotation.RotationX,
        "y": Rotation.RotationY,
        "z": Rotation.RotationZ,
    }

    r0 = axis_to_rotation[order[0]](angles[..., 0])
    r1 = axis_to_rotation[order[1]](angles[..., 1])
    r2 = axis_to_rotation[order[2]](angles[..., 2])

    return Tensor.MatMul(r0, Tensor.MatMul(r1, r2))


class BVH:
    def __init__(self, path, scale=1.0):
        self._path = path
        self._scale = scale

        if not os.path.isfile(path):
            raise FileNotFoundError(f"BVH file not found: {path}")

        f = open(path, "r")

        i = 0
        active = -1
        end_site = False

        names = []
        offsets = np.array([], dtype=np.float32).reshape((0, 3))
        parents = np.array([], dtype=int)

        channels = None
        order = None
        framerate = None

        for line in f:
            if "HIERARCHY" in line:
                continue
            if "MOTION" in line:
                continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]], dtype=np.float32), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue

            if "{" in line:
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(
                r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
            )
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))], dtype=np.float32)
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis : 2 + channelie]
                    if all(p in channelmap for p in parts):
                        order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]], dtype=np.float32), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros((fnum, len(names), 3), dtype=np.float32)
                continue

            fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                framerate = float(fmatch.group(1))
                continue

            dmatch = line.strip().split(" ")
            if dmatch:
                data_block = np.array(list(map(float, dmatch)), dtype=np.float32)
                N = len(parents)
                fi = i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        if order is None:
            raise ValueError(f"Could not detect rotation order from BVH file: {path}")
        if framerate is None:
            raise ValueError(f"Could not detect frame time from BVH file: {path}")

        self._names = names
        self._parents = parents
        self._offsets = offsets
        self._positions = positions
        self._rotations = rotations
        self._order = order
        self._framerate = 1.0 / framerate
        self._channels = channels

    @property
    def Filename(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    def FindParent(self, name, candidates):
        idx = self._names.index(name)
        parent_idx = self._parents[idx]
        while parent_idx != -1:
            if self._names[parent_idx] in candidates:
                return self._names[parent_idx]
            parent_idx = self._parents[parent_idx]
        return None

    def LoadMotion(self, names=None, floor=None) -> Motion:
        num_frames = self._rotations.shape[0]
        num_joints = self._rotations.shape[1]

        rotation_matrices = _euler_to_rotation_matrix(self._rotations, self._order)
        local_positions = Tensor.Create(self._positions)
        local_positions = self._scale * local_positions
        local_matrices = Transform.TR(local_positions, rotation_matrices)

        global_matrices = np.zeros((num_frames, num_joints, 4, 4), dtype=np.float32)
        for joint_idx in range(num_joints):
            parent_idx = self._parents[joint_idx]
            if parent_idx == -1:
                global_matrices[:, joint_idx] = local_matrices[:, joint_idx]
            else:
                global_matrices[:, joint_idx] = Transform.Multiply(
                    global_matrices[:, parent_idx], local_matrices[:, joint_idx]
                )

        all_parent_names = []
        for parent_idx in self._parents:
            if parent_idx == -1:
                all_parent_names.append(None)
            else:
                all_parent_names.append(self._names[parent_idx])

        if names is None:
            hierarchy = Hierarchy(bone_names=self._names, parent_names=all_parent_names)
            frames = global_matrices
        else:
            parent_names = [self.FindParent(name, names) for name in names]
            hierarchy = Hierarchy(bone_names=names, parent_names=parent_names)
            name_to_index = {name: i for i, name in enumerate(self._names)}
            indices = [name_to_index[name] for name in names if name in name_to_index]
            frames = global_matrices[:, indices]

        if floor is not None:
            if floor not in self._names:
                print(
                    f"Floor node '{floor}' not found in BVH file. Available nodes: {self._names}"
                )
            else:
                floor_idx = self._names.index(floor)
                offset = global_matrices[:, floor_idx]
                frames = Transform.TransformationTo(frames, offset.reshape(-1, 1, 4, 4))

        return Motion(
            name=self.Filename,
            hierarchy=hierarchy,
            frames=frames,
            framerate=self._framerate,
        )
