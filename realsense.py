import realsense2
import slam
import tempfile
import time
import json
import os
import numpy as np
import transformation
import quaternion

class RealsenseCamera:
    def __init__(self, parameter_file, vocabulary_file):
        with open(parameter_file) as f:
            self.parameters = json.load(f)
        self.vocabulary_file = vocabulary_file
        self.template = """%YAML:1.0

        Camera.fx: {fx}
        Camera.fy: {fy}
        Camera.cx: {cx}
        Camera.cy: {cy}

        Camera.k1: {k1}
        Camera.k2: {k2}
        Camera.p1: {p1}
        Camera.p2: {p2}
        Camera.k3: {k3}

        Camera.width: {width}
        Camera.height: {height}

        Camera.fps: {fps}

        Camera.bf: {bf}

        Camera.RGB: 1

        ThDepth: {thdepth}

        DepthMapFactor: 5000.0

        ORBextractor.nFeatures: {nfeatures}
        ORBextractor.scaleFactor: 1.2
        ORBextractor.nLevels: 8
        ORBextractor.iniThFAST: 20
        ORBextractor.minThFAST: 7"""

    def initialize(self):
        config_contents = self.template.format(**self.parameters)
        fd, config_file = tempfile.mkstemp()
        os.write(fd, bytes(config_contents, "UTF-8"))
        os.write(fd, bytes(os.linesep, "UTF-8"))
        os.close(fd)
        self.slam = slam.SLAM(self.vocabulary_file, config_file)
        height = int(self.parameters["height"])
        width = int(self.parameters["width"])
        fps = int(self.parameters["fps"])
        self.realsense = realsense2.RealsenseReader(height, width, fps)

    def get_scene(self):
        frameset = self.realsense.get_frames()
        t = time.time() - self.parameters["delay"]
        pose = np.array(self.slam.update(frameset.color, frameset.depth, t))
        color = np.array(frameset.color).astype("uint8")
        points = np.array(frameset.vertices).reshape(color.shape)

        if pose.size == 1:
            return t, None, points, color
        else:
            pose = pose[:3,:]
            pose[:,3] *= 5.08
            return t, transformation.from_pose_matrix(pose).inverse(), points, color