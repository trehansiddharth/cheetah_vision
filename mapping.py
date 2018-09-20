import numpy as np
import scipy.ndimage

class Subspace:
    """
    Stores information on the basis vectors, normal vectors, and origin forming a subspace
    to simplify the projection and deprojection of vectors within the subspace.
    """
    def __init__(self, T, Tinv, R, Rinv, origin):
        """
        Costruct a subspace given vectors and matrices defining it. Should not be used
        directly.
        """
        self.T = T
        self.Tinv = Tinv
        self.R = R
        self.Rinv = Rinv
        self.origin = origin

    @staticmethod
    def from_components(basis, normal, origin=0):
        """
        Construct an n-dimensional subspace within a k-dimensional superspace given a set
        of basis vectors, a set of normal vectors, and a point within the subspace.

        Args:
            basis: (n, k) np.array: Set of k-dimensional vectors defining the basis vectors
                for the supspace. The vectors do not have to be unitary.
            normal: (m, k) np.array: Set of k-dimensional vectors defining the normal
                vectors for the subspace. Must be orthogonal to the basis vectors. The
                vectors do not have to be unitary.
            origin: (k,) np.array (optional): A k-dimensional point on the subspace. If not
                given, it is assumed the subspace contains the zero vector as the origin.
        """
        T = basis
        Tinv = np.linalg.pinv(basis)
        R = normal
        Rinv = np.linalg.pinv(normal)
        origin = origin
        return Subspace(T, Tinv, R, Rinv, origin)

    def _parameters(self):
        return (self.T, self.Tinv, self.R, self.Rinv, self.origin)

    def project(self, points, method="orthographic"):
        """
        Projects a point from a higher k-dimensional superspace to the n-dimensional
        subspace.

        Args:
            points: (p, k) np.array: Set of k-dimensional points to project.
            method: "orthographic" or "perspective" (optional): Whether to perform an
                orthographic or perspective projection. Default is "orthographic".
        
        Returns: 2-tuple:
            images: (p, n) np.array: n-dimensional images of the points in the subspace.
            residuals: (p, k) np.array: k-dimensional residual vectors calculated by
                projecting the points onto the normal vectors to the subspace.
        """
        if method == "orthographic":
            images = (points - self.origin) @ self.Tinv
            residuals = (points - self.origin) @ self.Rinv
        elif method == "perspective":
            residuals = (points - self.origin) @ self.Rinv
            images = ((points - self.origin) @ self.Tinv) / residuals
        return images, residuals
    
    def project_vector(self, vector):
        """
        Projects a vector (indicating a direction rather than a position) from a higher
        k-dimensional superspace to the n-dimensional subspace.

        Args:
            vector: (p, k) np.array: Set of k-dimensional vectors to project.
        
        Returns:
            projection: (p, n) np.array: Projection of the vector onto the subspace.
        """
        return vector @ self.Tinv

    def deproject(self, images, residuals=None, method="orthographic"):
        """
        The inverse of a projection: takes a n-dimensional image in the subspace and
        (optionally) a residual indicating its "out-of-plane" component, and returns
        the corresponding point in the k-dimensional superspace

        Args:
            images: (p, n) np.array: Set of n-dimensional points in the subspace to
                deproject.
            residuals: (p, k) np.array (optional): Set of k-dimensional residuals to add
                to the deprojected vectors.
            method: "orthographic" or "perspective" (optional): Whether to perform an
                orthographic or perspective deprojection. Default is "orthographic".
        
        Returns:
            points: (p, k) np.array: The set of k-dimensional points that, when projected,
                will produce the provided images and residuals.
        """
        if method == "orthographic":
            inplane = images @ self.T + self.origin
            outplane = 0 if residuals is None else residuals @ self.R
        elif method == "perspective":
            outplane = 0 if residuals is None else residuals @ self.R
            depths = 1 if residuals is None else np.linalg.norm(outplane, axis=1).reshape(-1, 1)
            inplane = depths * (images @ self.T) + self.origin
        return inplane + outplane

    def accomodate(self, points, method="orthographic"):
        """
        Creates a lattice with an array large enough to accommodate a given set of points.

        Args:
            points: (p, k) np.array: Set of k-dimensional points in the superspace that
                the lattice should be large enough to accommodate.
            method: "orthographic" or "perspective" (optional): Whether to perform an
                orthographic or perspective projection. Default is "orthographic".
        
        Returns:
            lattice: Lattice: The lattice with an empty array large enough to accommodate
                all the provided points.
        """
        points_T, residuals = self.project(points, method=method)
        offset = points_T.min(axis=0)
        terminus = points_T.max(axis=0)
        shape = (terminus - offset).astype("int") + 3

        subspace = Subspace(*self._parameters())

        return Lattice(np.empty(shape), subspace, offset.astype("int"))

class Lattice(Subspace):
    """
    A combination of a numpy array that can store a value at different cells and a
    Subspace that can map numpy indices to physical points. This can be used,
    for example, to store an occupancy grid of a physical environment without having to
    worry about which position in the array corresponds to which point in the real world.
    """
    def __init__(self, arr, subspace, offset=0, default=np.nan):
        """
        Constructs a lattice from an array, a subspace, and an offset parameter describing
        how the origin of the array maps to the subspace.

        Args:
            arr: n-dimensional np.array: Array to store data in.
            subspace: Subspace: n-dimensional subspace that the array represents a map of.
            offset: (n,) np.array (optional): The physical point corresponding to the
                zero-index of the array. This is necessary because an array can't store
                data in negative indices. The default offset is 0 (origin of array
                corresponds to origin of subspace).
        """
        Subspace.__init__(self, *subspace._parameters())
        self.arr = arr
        # self.offset = tuple(self.arr.ndim * [0]) if offset == 0 else offset
        self.offset = offset
        self.default = default
    
    # def resize(self, slc):        
    #     slices = self.arr.ndim * [slice(None, None, None)
    #     if (type(slc) == int) or (type(slc) == slice) or \
    #         (type(slc) = np.ndarray and slc.dtype == "bool"):
    #         slc = (slc,)
        
    #     new_shape = list(self.arr.shape)
    #     new_offset = list(self.offset)
    #     copy_slice = self.arr.ndim * [None]
    #     for i, s in enumerate(slc):
    #         if type(s) == int:
    #             smallest = s + self.offset
    #             largest = s + self.offset
    #             slices[i] = s + self.offset
    #         elif hasattr(s, "__iter__"):
    #             is_mask = False
    #             if type(s) == np.ndarray:
    #                 if s.dtype == "bool":
    #                     is_mask = True
    #                     slices[i] = s
    #                 else:
    #                     slices[i] = s + self.offset
    #             else:
    #                 slices[i] = [t + self.offset for t in s]
    #             smallest = 0 if is_mask else min(s) + self.offset
    #             largest = 0 if is_mask else max(s) + self.offset
    #         elif type(s) == slice:
    #             start = s.start and s.start + self.offset
    #             stop = s.stop and s.stop + self.offset
    #             step = s.step
    #             smallest = min(start, stop)
    #             largest = max(start, stop)
    #             slices[i] = slice(start, stop, step)
    #         else:
    #             raise ValueError
            
    #         if smallest < 0:
    #             new_shape[i] += self.shape[i]
    #             new_offset[i] += self.shape[i]
    #             copy_slice[i] = slice(0, self.shape[i])
    #         elif largest >= self.shape[i]:
    #             newshape[i] += self.shape[i]
    #             copy_slice[i] = slice(self.shape[i], None)
        
    #     if new_shape != self.shape:
    #         new_arr = self.default * np.ones(tuple(new_shape))
    #         new_arr[tuple(copy_slice)] = self.arr
    #         self.arr = new_arr
    #         self.offset = tuple(new_offset)
        
    #     return tuple(slices)

    def __getitem__(self, slc):
        """
        Gets array values indexed by an np.array of physical coordiates in the subspace.

        Args:
            slc: (p, n) np.array: Array of points in the n-dimensional subspace to get
                values of. Hint: if you have a (p, k) array of points in the superspace,
                project it into the subspace first. Must be integer type (use "round").
        
        Returns:
            values: (p,) np.array: Array of values in the array at the indices
                corresponding to slc.
        """
        return self.arr[tuple(slc.transpose() - self.offset)]
        #return self.arr[self.resize(slc)]

    def __setitem__(self, slc, values):
        """
        Sets array values indexed by an np.array of physical coordinates in the subspace
        to the values provided.

        Args:
            slc: (p, n) np.array: Array of points in the n-dimensional subspace to set
                values of. Hint: if you have a (p, k) array of points in the superspace,
                project it into the subspace first. Must be integer type (use "round").
            values: (p,) np.array: Array of values to set the corresponding indices to.
        """
        self.arr[tuple(slc.transpose() - self.offset)] = values
        #self.arr[self.resize(slc)] = values
    
    def filter(self, indices):
        """
        Runs a filter over an array of indices to keep only the ones that fall within
        the bounds of the array.

        Args:
            indices: (p, n) np.array: Array of indices to filter.
        
        Returns:
            indices_: (q, n) np.array: Array of indices from the original array that
                contains only the ones that fall within the bounds of the array.
        """
        return indices[self.argfilter(indices)]
    
    def argfilter(self, indices):
        """
        Same as "filter" but returns a boolean array of whether each index falls within
        the bounds of the array or not.

        Args:
            indices: (p, n) np.array: Array of indices to test.
        
        Returns:
            mask: (p,) np.array: Boolean array indicating whether each index is within
                the bounds of the lattice of not.
        """
        return np.all(indices - self.offset > 0, axis=1) & \
            np.all(indices - self.offset < self.arr.shape, axis=1)

    def round(self, indices):
        """
        Rounds each index to the nearest whole value. Turns an array of physical
        coordinates in the subspace to its corresponding index in the array. Does not
        care about whether an index falls within the bounds of the array or not (use
        "filter" for that).

        Args:
            indices: (p, n) np.array: Array of indices to round.
        
        Returns:
            indices_: (p, n) np.array: Integer array of rounded indices.
        """
        return np.around(indices).astype("int")

    def indices(self):
        """
        All the valid indices in in the array, compensated for by the offset.

        Returns:
            indices: (p, n) np.array: Array of all indices that can be used to validly
                index the lattice.
        """
        return np.array(list(np.ndindex(self.arr.shape))) + self.offset

    def points(self):
        """
        All the valid points in the superspace of the array (just a deprojection
        of the indices).

        Returns:
            points: (p, k) np.array: Array of all the points that map to a valid index
                in the lattice.
        """
        return self.deproject(self.indices())

def depth_transform(points, subspace, method="orthographic"):
    """
    Projects and discretizes a set of points by projecting them onto a lattice defined
    by a subspace, and if multiple points fall onto the same lattice points, it keeps
    only the one that is closer in depth (smaller residual). Depending on the type of
    projection, it typically produces a "shadow" of the points closest to a particular
    point or line.

    Args:
        points: (p, k): Set of points in a k-dimensional spaces for which to compute
            the depth transform.
        subspace: Subspace: n-dimensional subspace that defines the projection for the
            shadow to be computed. The length of the basis vectors of the subspace define
            the discreteness of the grid onto which the points are projected. When
            multiple points map to the same discretized grid point, the one with the
            smallest residual vector upon projection is kept.
        method: "orthographic" or "perspective" (optional): Whether to perform an
            orthographic or perspective projection. Default is "orthographic".
    
    Returns:
        lattice: Lattice: A lattice on the subspace provided, where each item in the
            array takes the value of the residual (the "depth" along the projection axis)
            to the original point. To reconstruct the points in the original superspace
            (so that only the "shadow" is kept), just deproject the lattice with the
            points being the indices of the lattice and the residuals being the array
            values.
    """
    lattice = subspace.accomodate(points, method=method)
    indices, depths = lattice.project(points, method=method)
    lattice.arr[:] = np.inf

    depths_argsort = np.argsort(-depths, axis=0).reshape(-1)
    depths_sorted = depths[depths_argsort]
    indices_sorted = lattice.round(indices[depths_argsort])

    lattice[indices_sorted] = depths_sorted.reshape(-1)
    return lattice

def minimum_lattice_distance(basis, px):
    """
    Assuming that points come from a grid of spacing px, the depth transform may not work
    very well if the grid spacing of the subspace onto which you are projecting is too
    small. In that case, points that are "behind" other points will be projected onto
    different grid points as ones that occlude them, and therefore will be kept when the
    depth transform is taken even though they are not part of the "shadow". This function
    determines the minimum lattice distance that allows for a valid distance transform
    given a particular grid spacing of the points and a particular basis of the subspace.

    Args:
        basis: (n, k) np.array: Basis vectors of the subspace onto which points will be
            projected in the depth transform. Each vector should be unitary.
        px: float: spacing between grid points (e.g. in the case of an image or voxel
            array, the size of a pixel).
    
    Returns:
        px_: float: minimum valid spacing between grid points of the subspace.
    """
    n, k = basis.shape
    diagonals = px * cartesian_product((np.array([1, -1]),) * n)
    return np.max(diagonals @ basis)

def update_heightmap(lattice, points):
    """
    Updates an existing heightmap mapping a subspace with a new set of points within a
    superspace.

    Args:
        lattice: Lattice: A lattice on an n-dimensional subspace where each value in the
            array represents the height of the tallest point (largest residual) that would
            be projected onto that position.
        points: (p, k) np.array: Points in the k-dimensional superspace to update the
            heightmap with.
    """
    indices, heights = lattice.project(points)
    indices = np.round(indices)
    heightmask = (heights.reshape(-1) > 0.1) & (heights.reshape(-1) < 4)
    heights = heights[heightmask]
    indices = indices[heightmask]
    boundsmask = lattice.argfilter(indices)
    heights = heights[boundsmask]
    indices = lattice.round(indices[boundsmask])
    lattice[indices] = heights.reshape(-1)

def filter_indices(subspace, points, camera, principal, hthresh=0, dthresh=3, cthresh=4):
    indices, heights = subspace.project(points)
    depths = (points - camera) @ principal.reshape(3, 1)
    mask = (heights.reshape(-1) > hthresh) & (heights.reshape(-1) < cthresh) & \
        (depths.reshape(-1) < dthresh)
    return indices[mask], heights[mask]

def update_occupancy_grid(lattice, indices):
    """
    Updates an existing occupancy grid mapping a subspace with a new set of points within
    a superspace.

    Args:
        lattice: Lattice: A lattice on an n-dimensional subspace where each value in the
            array represents whether or not there is a point that would be projected onto
            that position.
        points: (p, k) np.array: Points in the k-dimensional superspace to update the
            occupancy grid with.
    """
    indices = lattice.filter(lattice.round(indices))
    lattice[indices] = 1

def update_occupancy_grid_weighted(lattice, points, camera, principal, hthresh=0):
    """
    Updates an existing occupancy grid mapping a subspace with a new set of points within
    a superspace.

    Args:
        lattice: Lattice: A lattice on an n-dimensional subspace where each value in the
            array represents whether or not there is a point that would be projected onto
            that position.
        points: (p, k) np.array: Points in the k-dimensional superspace to update the
            occupancy grid with.
    """
    indices, heights = lattice.project(points)
    depths = (points - camera) @ principal.reshape(3, 1)
    indices = indices[(heights.reshape(-1) > hthresh) & (heights.reshape(-1) < 4) & \
        (depths.reshape(-1) < 3.0)]
    indices = lattice.filter(lattice.round(indices))
    lattice[indices] += 1

def update_occupancy_grid_shadowed(lattice, points, camera, principal, fpx, hthresh=0):
    """
    Updates an existing occupancy grid mapping a subspace with a new set of points within
    a superspace, but first runs the depth transform on the points to pick only the ones
    that form a "shadow" with respect to the camera position and orientation. Useful to
    remove certain sources of noise within pointclouds.

    Args:
        lattice: Lattice: A lattice on an n-dimensional subspace where each value in the
            array represents whether or not there is a point that would be projected onto
            that position (however points that are not shadows are considered noise).
        points: (p, k) np.array: Points in the k-dimensional superspace to update the
            occupancy grid with. A depth transform is run on this to filter out noise.
    """
    indices, heights = lattice.project(points)
    depths = (points - camera) @ principal.reshape(3, 1)
    indices = indices[(heights.reshape(-1) > hthresh) & (heights.reshape(-1) < 4.0) & \
        (depths.reshape(-1) < 3.0)]
    if len(indices) > 0:
        tangential = np.cross(lattice.R[0], principal)
        principal_proj = lattice.project_vector(principal).reshape(1, -1)
        principal_proj = principal_proj / np.linalg.norm(principal_proj)
        tangential_proj = lattice.project_vector(tangential).reshape(1, -1)
        tangential_proj = tangential_proj / (np.linalg.norm(tangential_proj) * fpx)
        camera_proj, _ = lattice.project(camera)
        subspace = Subspace.from_components(tangential_proj, principal_proj, camera_proj)
        line = depth_transform(indices, subspace, method="perspective")
        mask = line.arr != np.inf
        shadow = line.deproject(line.indices()[mask], line.arr.reshape(-1, 1)[mask], method="perspective")
        shadow = lattice.filter(lattice.round(shadow))
        lattice_points = np.array(np.where(lattice.arr)).transpose()
        lattice_points = filter_occluded(lattice_points, line, method="perspective")
        lattice.arr[:] = 0
        lattice[lattice_points] = 1
        lattice[shadow] = 1

def filter_occluded(points, depth_lattice, method="orthogonal"):
    indices, depths = depth_lattice.project(points, method=method)
    indices = depth_lattice.round(indices)
    within_fov = depth_lattice.argfilter(indices)
    within_fov_points = points[within_fov]
    within_fov_indices = indices[within_fov]
    within_fov_points_depths = depths[within_fov].reshape(-1)
    within_fov_ref_depths = depth_lattice[within_fov_indices]
    valid_within_fov = (within_fov_points_depths > within_fov_ref_depths) \
        | (within_fov_points_depths < 0)
    valid_within_fov_points = within_fov_points[valid_within_fov]
    outside_fov_points = points[~within_fov]
    if len(valid_within_fov_points) == 0:
        return outside_fov_points
    elif len(outside_fov_points) == 0:
        return valid_within_fov_points
    else:
        return np.vstack([valid_within_fov_points, outside_fov_points])

def normed_convolution(array, kernel):
    """
    Convolves an image with a kernel, but values in the array that are np.inf are ont
    considered-- the convolution is normalized considering how many values in a window
    are actually valid values.

    Args:
        array: n-dimensional np.array: Array to run the convolution on.
        kernel: n-dimensional np.array: Kernel to convolve with.
    
    Returns:
        array_: n-dimensional np.array: Result of the normed convolution.
    """
    zeros = array.copy()
    zeros[array == np.inf] = 0
    ones = np.ones_like(array)
    ones[array == np.inf] = 0
    unnormed = scipy.ndimage.filters.convolve(zeros, kernel)
    population = scipy.ndimage.filters.convolve(ones, kernel)
    return unnormed / population

def interpolate_array(array, gamma=1.0, cutoff=5):
    """
    Interpolates values in an array that are missing (indicated by a value of np.inf) by
    running an approximately nearest neighbor approach within a certain distance cutoff.

    Args:
        array: n-dimensional np.array: Array to interpolate.
        gamma: float (optional): Controls the degree to which nearest neighbors are
            weighted relative to next-nearest neighbors and so on. The larger this value
            is, the closer this algorithm is to true nearest-neighbors, but also the more
            succeptible it is to overflow. Default value is 1.0.
        cutoff: int (optional): Actually the window size of the convolution kernel, but can
            be interpreted as the maximum possible pixel distance to a point for it to be
            considered a neighbor. The smaller it is, the faster the runtime. Default value
            is 5.
    
    Returns:
        interpolated: n-dimensional np.array: Interpolated array.
    """
    dims = len(array.shape)
    Xs = np.meshgrid(*([np.arange(-cutoff, cutoff + 1)] * dims))
    kernel = np.exp(-gamma * np.linalg.norm(Xs, axis=0))
    smoothed = normed_convolution(array, kernel)
    return np.where(array == np.inf, smoothed, array)