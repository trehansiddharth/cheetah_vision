# TODO

1. Support for poses and pointclouds in multiple coordinate systems:
    * "frame" parameter in pose_t and pointcloud_t type
    * Process for transformation determination and messaging over TCP
    * "frames.txt" file for set of coordinate basis transformations
2. Conversion of cheetah_state_t to pose_t:
    * Separate process for conversion?
    * Built-in function for conversion?
    * Hard-code different type?
3. Planner to command cheetah velocity
