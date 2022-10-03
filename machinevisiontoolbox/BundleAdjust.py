import time
import numpy as np
import scipy as sp

try:
    import pgraph
    pgraph_installed = True
except:
    print('pgraph not installed')
    pgraph_installed = False
from spatialmath import base
from spatialmath import SE3, SO3, UnitQuaternion
import matplotlib.pyplot as plt

from machinevisiontoolbox import CentralCamera

# We use the PGraph graph package and subclass the nodes and edges for an
# undirected graph
#
# Each node has an index into the state vector given by its index and index2
# properties.  These are initialized by a call to update_index

if pgraph_installed:
    class _Common:
        @property
        def index(self):
            """
            Index into the state vector (base method)

            :return: the index of the start of this object's state in the state vector
            :rtype: int
            """
            return self._index

        @property
        def index2(self):
            """
            Index into the variable state vector (base method)

            :return: the index of the start of this object's state in the variable state vector
            :rtype: int
            """
            return self._index2

        @property
        def isfixed(self):
            """
            Value is fixed (base method)

            :return: Quantity is fixed
            :rtype: bool

            This viewpoint or landmark will not be adjusted during optimization.
            """
            return self._fixed

    class ViewPoint(pgraph.UVertex, _Common):

        def __init__(self, x, fixed=False, color=None):
            """
            Create new camera viewpoint

            :param x: viewpoint pose as translation + vector part of unit quaternion
            :type x: array_like(6)
            :param fixed: camera is fixed, defaults to False
            :type fixed: bool, optional
            :param color: color with which to draw camera icon, defaults to None
            :type color: str, optional

            Represent a camera viewpoint in the bundle adjustment problem.  If
            the camera is not ``fixed`` it will be adjusted during the
            optimization.

            :seealso: :class:`PGraph.UVertex`
            """
            super().__init__()

            self.coord = x    
            self._fixed = fixed
            self._color = color

        @property
        def pose(self):
            """
            Get pose of camera

            :return: pose as an SE(3)
            :rtype: :class:`~spatialmath.pose3d.SE3`
            """
            t = self.coord[:3]
            qv = self.coord[3:]

            return SE3(t) * UnitQuaternion.Vec3(qv).SE3()

    class Landmark(pgraph.UVertex, _Common):
        def __init__(self, P, fixed=False):
            """
            Create new landmark point

            :param P: landmark coordinate
            :type P: ndarray(3)
            :param fixed: point is fixed, defaults to False
            :type fixed: bool, optional

            Represent a world point in the bundle adjustment problem.  If the
            point is not ``fixed`` it will be adjusted during the optimization.
            
            :seealso: :meth:`pgraph.UVertex`
            """
            super().__init__()
            self._P = P
            self._fixed = fixed

        @property
        def P(self):
            """
            Get landmark position

            :return: landmark position in 3D
            :rtype: ndarray(3)
            """
            return self._P
    class Observation(pgraph.Edge):
        def __init__(self, camera, landmark, uv):
            """
            Create new landmark observation

            :param camera: the camera that made the observation
            :type camera: :class:`ViewPoint`
            :param landmark: the observed landmark
            :type landmark: :class:`Landmark`
            :param uv: the image plane coordinates of the observed landmark
            :type uv: ndarray(2)

            Represent the observation of a point by a camera in the bundle
            adjustment problem.
            
            :seealso: :meth:`pgraph.Edge`
            """
            super().__init__(camera, landmark, cost=0)
            self._p = uv

        @property
        def p(self):
            """
            Get image plane projection

            :return: observed projection of landmark on image plane
            :rtype: ndarray(2)
            """

            return self._p
    class BundleAdjust:

        def __init__(self, camera):
            r"""
            Create a bundle adjustment problem

            :param camera: model of the moving camera
            :type camera: CentralCamera instance

            Implementation of a workable, easy to follow, but simplistic, bundle
            adjustment algorithm.

            It uses SciPy sparse linear algebra functions to solve the update
            equation. The state vector comprises, in order:

            - a 6-vector for every view, the camera pose :math:`(t_x, t_y, t_z,
              q_x, q_y, q_z)` as a translation vector and the vector part of the
              unit quaternion
            - a 3-vector for every landmark, (X, Y, Z)

            Cameras and landmarks can be fixed in which case we have a variable state
            vector, shorter than the state vector, holding only the states corresponding
            to movable cameras and landmarks.


            .. warning:: This class assumes that all camera views have the same camera
                intrinsics.

            :reference:
                - Robotics, Vision & Control for Python, Section 14.3.2, 
                  P. Corke, Springer 2023.

            :seealso: :meth:`optimize` :class:`~machinevisiontoolbox.Camera.CentralCamera` :class:`pgraph.UGraph`
            """
            self.camera = camera

            # we use a PGraph object to represent nodes and edges as an undirected graph:
            #  - ViewPoint class for vertices representing camera poses as (tx, ty,
            #    tz, qx, qy, qz)
            #  - Landmark for  vertices representing landmark positions as (x,
            #    y, z)
            #  - Observation for edges representing the observation of a landmark by
            #    a view
            #
            self.g = pgraph.UGraph(6)  # initialize the graph, nodes have 6D coordinates

            self._nviews = 0  #number of cameras
            self._nlandmarks = 0 #number of landmark points
            self._nvarstate = 0

            self.views = []           # list of view nodes
            self.landmarks = []       # list of landmark nodes
            
            self.fixedviews = []      # list of view nodes that are fixed
            self.fixedlandmarks = []  # list of landmark nodes that are fixed

            self.index_valid = False

        def update_index(self):
            if self.index_valid:
                return

            index = 0
            index2 = 0
            for i, view in enumerate(self.views):
                view.id = i
                view._index = index
                view._index2 = index2
                index += 6
                if not view.isfixed:
                    index2 += 6

            for i, landmark in enumerate(self.landmarks):
                landmark.id = i
                landmark._index = index
                landmark._index2 = index2
                index += 3
                if not landmark.isfixed:
                    index2 += 3

        @property
        def nviews(self):
            """
            Number of camera views

            :return: Number of camera views
            :rtype: int

            :seealso: :meth:`add_view`
            """
            return self._nviews

        @property
        def nlandmarks(self):
            """
            Number of landmarks

            :return: Number of landmarks
            :rtype: int

            :seealso: :meth:`add_landmark`
            """
            return self._nlandmarks

        @property
        def nstates(self):
            """
            Length of state vector

            :return: Length of the state vector
            :rtype: int

            This includes fixed views and landmarks whose state will not be
            updated in the opimization.

            :seealso: :meth:`nvarstate`
            """
            return  6 * self.nviews + 3 * self.nlandmarks

        @property
        def nvarstates(self):
            """
            Length of variable state vector

            :return: Length of the variable state vector
            :rtype: int

            This is the length of the subset of the state vector that excludes
            fixed views and landmarks. It only includes cameras and landmarks
            whose state will be updated in the opimization.

            :seealso: :meth:`nstates`
            """
            return  6 * (self.nviews - len(self.fixedviews)) \
                + 3 * (self.nlandmarks - len(self.fixedlandmarks))

        def add_view(self, pose, fixed=False, color='black'):
            """
            Add camera view to bundle adjustment problem

            :param pose: camera pose
            :type pose: :class:`~spatialmath.pose3d.SE3`,  array_like(7)
            :param fixed: the camera is fixed, defaults to False
            :type fixed: bool, optional
            :return: new camera viewpoint
            :rtype: :class:`ViewPoint`
            
            Creates a camera node and adds it to the bundle adjustment problem.
            
            The camera ``pose``  can be :class:`~spatialmath.pose3d.SE3` or a vector (1x7)
            comprising translation and unit quaternion in vector form.

            If the camera is fixed (anchored) it will not be adjusted in the
            optimization process.
            
            .. note:: Adds a :class:`ViewPoint` object as a node in the 
                underlying scene graph.

            :seealso: :meth:`add_landmark` :meth:`add_projection`
            """
            if isinstance(pose, SE3):
                t = pose.t
                q = base.r2q(pose.R)
            else:
                base.assertvector(pose, 7)
                q = pose[:4]
                t = pose[4:]
            if q[0] < 0:
                q = -q
            x = np.r_[t, q[1:]]

            v = ViewPoint(x, fixed=fixed, color=color)
            v.name = f"view#{self._nviews}"
            self._nviews += 1

            self.g.add_vertex(v)
            self.views.append(v)
            if fixed:
                self.fixedviews.append(v)
            v.ba = self # back reference to the BA problem
            self.index_valid = False
            return v

        def add_landmark(self, P, fixed=False):
            """
            Add 3D landmark point to bundle adjustment problem

            :param P: 3D world point, aka landmark
            :type P: array_like(3)
            :param fixed: the landmark is fixed, defaults to False
            :type fixed: bool, optional
            :return: new landmark
            :rtype: :class:`Landmark` instance

            Create a landmark node and add it to the bundle adjustment problem.

            If the landmark is fixed (anchored) it will not be adjusted in the
            optimization process.

            .. note:: Adds a :class:`Landmark` object as a node in the 
                underlying scene graph.

            :seealso: :meth:`add_view` :meth:`add_projection` 
            """
            base.assertvector(P, 3)
            
            # P = np.r_[P, 0, 0, 0]

            l = Landmark(P, fixed=fixed)
            l.name = f"landmark#{self._nlandmarks}"
            l.coord = P
            self._nlandmarks += 1
            self.g.add_vertex(l)
            self.landmarks.append(l)
            if fixed:
                self.fixedlandmarks.append(c)
            l.ba = self # back reference to the BA problem
            self.index_valid = False
            return l

        def add_projection(self, viewpoint, landmark, uv):
            """
            Add camera observation to bundle adjustment problem

            :param view: camera viewpoint
            :type view: :class:`ViewPoint`
            :param landmark: landmark point
            :type landmark: :class:`Landmark`
            :param uv: image plane coordinate
            :type uv: array_like(2)
            
            Add an observation by ``viewpoint`` of a ``landmark`` to the bundle
            adjustment problem.  

            .. note:: Adds a :class:`Observation` object as an edge in the
                underlying scene graph.

            :seealso: :meth:`add_view` :meth:`add_landmark`
            """
            assert len(uv) == 2, 'uv must be a 2-vector'
            
            edge = Observation(viewpoint, landmark, uv.flatten()) # create edge object
            e = viewpoint.connect(landmark, edge=edge)  # connect nodes with it
            e.name = viewpoint.name + "--" + landmark.name

        @classmethod
        def load_SBA(cls, cameraFile, pointFile, calibFile, imagesize=None):
            """
            Load bundle adjustment data files

            :param cameraFile: name of file with camera view data
            :type cameraFile: str
            :param pointFile: name of file with landmark data
            :type pointFile: str
            :param calibFile: name of file with camera intrinsic data
            :type calibFile: str
            :param imagesize: image plane dimensions in pixels, if not given infer
                it from principal point data in ``calibFile``
            :type imagesize: array_like(2)

            Provides access to bundle adjustment problems from data files as distributed with the SBA package.
            Details of the file format are given in the source code comments.
            
            Example:
            
            To solve the 7-point bundle adjustment problem distributed with
            SBA 1.6::
            
                >>> ba = Bundle.load_SBA('7cams.txt', '7pts.txt', 'calib.txt')
                >>> X = ba.optimize()

            :reference:
                - Sparse Bundle Adjustment package by Manolis Lourakis,
                  http://users.ics.forth.gr/~lourakis/sba
            
            :seealso: :meth:`add_view` :meth:`add_landmark` :meth:`add_projection`
            """
            # Adopted from sba-1.6/matlab/eucsbademo.m 

            # read calibration parameters
            #
            # f/rho_u  skew      u0
            # 0        f/rho_v   v0
            # 0        0         1  
            K = np.loadtxt(calibFile)

            # create the camera object
            if imagesize is None:
                # no image plane size given
                # infer it from the principal point
                imagesize = 2 * K[:2, 2]

            camera = CentralCamera(
                f=K[0, 0],
                rho = [1, K[0, 0] / K[1, 1]],
                pp = K[:2, 2],
                imagesize = imagesize
            )

            # create a bundle adjustment instance
            ba = cls(camera)

            # read camera views
            #
            # each line is: qs qx qy qz tx ty tz 
            for pose in np.loadtxt(cameraFile):
                ba.add_view(pose)

            # read points and projections
            # 
            # The lines are of the form:
            #
            # X Y Z  NFRAMES  FRAME0 x0 y0  FRAME1 x1 y1 ...
            #
            # corresponding to a single 3D point and multiple projections:
            # 
            # - X, Y, Z is the points' Euclidean 3D coordinates,
            # - NFRAMES the total number of camera views in which the point is
            #   visible and there will follow NFRAMES subsequent triplets 
            # - FRAME x y specifies that the 3D point in question projects to pixel
            #   (x, y) in view number FRAME. 
            # 
            # For example, the line:
            #
            # 100.0 200.0 300.0 3  2 270.0 114.1 4 234.2 321.7 5 173.6 425.8
            #
            # describes a world point (100.0, 200.0, 300.0) that is visible in 
            # three views: view 2 at (270.0, 114.1), view 4 at (234.2, 321.7) and
            # view 5 at (173.6, 425.8)
            with open(pointFile, 'r') as file:
                npts = 0
                for line in file:
                    if len(line) == 0 or line[0] == '#':
                        continue

                    data = line.split()

                    #read X, Y, Z, nframes
                    P = np.array([float(x) for x in data[:3]])
                    data = data[3:]
                    npts += 1
                
                    #create a node for this point
                    landmark = ba.add_landmark(P)
                    
                    #now find which cameras it was seen by
                    nframes = int(data.pop(0))
                    for i in range(nframes):  #read "nframes" id, x, y triplets
                        id = int(data.pop(0))
                        u = float(data.pop(0))
                        v = float(data.pop(0))

                        #add a landmark projection
                        ba.add_projection(ba.views[id], landmark, np.r_[u, v])
            return ba
        
        # =============== METHODS TO SOLVE PROBLEMS ==================== #
        
        def optimize(self, x=None, animate=False, lmbda=0.1, 
            lmbdamin=1e-8, dxmin=1e-4, tol=0.5, iterations=1000, verbose=False):
            """
            Perform the bundle adjustment

            :param x: state vector, defaults to the state vector in the instance
            :type Xx: ndarray(N), optional
            :param animate: graphically animate the updates, defaults to False
            :type animate: bool, optional
            :param lmbda: initial damping term, defaults to 0.1
            :type lmbda: float, optional
            :param lmbdamin: minimum value of ``lmbda``, defaults to 1e-8
            :type lmbdamin: float, optional
            :param dxmin: terminate optimization if state update norm falls below this 
                threshold, defaults to 1e-4
            :type dxmin: float, optional
            :param tol: terminate optimization if error total reprojection error
                falls below this threshold, defaults to 0.5 pixels
            :type tol: float, optional
            :param iterations: maximum number of iterations, defaults to 1000
            :type iterations: int, optional
            :param verbose: show Levenberg-Marquadt status, defaults to False
            :type verbose: bool, optional
            :return: optimized state vector
            :rtype: ndarray(N)

            Performs a Levenberg-Marquadt style optimization of the bundle
            adjustment problem which repeatedly calls :meth:`solve`.  Adjusts
            camera poses and landmark positions in order to minimize the total
            reprojection error.

            :reference:
                - Robotics, Vision & Control for Python, Section 14.3.2, 
                  P. Corke, Springer 2023.

            :seealso: :meth:`nstates` :meth:`solve` :meth:`build_linear_system`
            """

            self.update_index()
            
            if x is None:
                x = self.getstate()
            x0 = x
            
            t0 = time.perf_counter()
            
            print(f"Bundle adjustment cost {self.errors(x0):.3g} -- initial")
            for i in range(iterations):
                if animate:
                    if not retain:
                        plt.clf()
                    g2.plot()
                    plt.pause(0.5)
                
                ta = time.perf_counter()
                # solve for the step
                dx, energy = self.solve(x, lmbda)

                # update the state
                x_new = self.updatestate(x, dx)

                # compute new value of cost
                enew = self.errors(x_new)
                
                dt = time.perf_counter() - ta
                print(f"Bundle adjustment cost {enew:.3g} (solved in {dt:.2f} sec)")
                # are we there yet?
                if enew < tol:
                    break
                
                # have we stopped moving
                if  base.norm(dx) < dxmin:
                    break

                # do the Levenberg-Marquadt thing, was it a good update?
                if enew < energy:
                    # step is accepted
                    x = x_new
                    if lmbda > lmbdamin:
                        lmbda /= np.sqrt(2)
                    if verbose:
                        print(f" -- step accepted: lambda = {lmbda:g}")
                else:
                    # step is rejected
                    lmbda *= 4
                    if verbose:
                        print(f" -- step rejected: lambda ={lmbda:g}")
            
            tf = time.perf_counter()
            err = np.sqrt(enew / self.g.ne)
            print(f"\n * {i + 1} iterations in {tf - t0:.1f} seconds")
            print(f" * Final RMS error is {err:.2f} pixels")
            
            return x_new, err
        
        def solve(self, x, lmbda=0.0):
            r"""
            Solve for state update

            :param x: state vector
            :type x: ndarray(N)
            :param lmbda: damping term, defaults to 0.0
            :type lmbda: float, optional
            :return: :math:`\delta \vec{X}`, update to the variable state vector
            :rtype: ndarray(M)

            Determines the state update :math:`\delta \vec{x}` by creating and
            solving the linear equation
            
            .. math:: \mat{H} \delta \vec{x} = \vec{b}
            
            where :math:`\mat{H}` is the Hessian and :math:`\mat{b}` is the the 
            projection error.

            .. note::
                - The damping term ``lmbda`` is added to the diagonal of the
                  Hessian to prevent problems when the Hessian is nearly
                  singular.
                - If the problem includes fixed cameras or landmarks then
                  :math:`\mbox{len}(\delta \vec{x}) < \mbox{len}(\vec{x})`
                  since fixed elements are omitted from the variable state
                  vector used for the optimization.

            :reference:
                - Robotics, Vision & Control for Python, Section 14.3.2, F.2.4, 
                  P. Corke, Springer 2023.

            :seealso: :meth:`build_linear_system`
            """
            # create the Hessian and error vector
            H, b, e = self.build_linear_system(x)
            
            # add damping term to the diagonal
            for i in range(self.nvarstates):
                H[i, i] += lmbda
            
            # solve for the state update
            #- could replace this with the Schur complement trick
            deltax = sp.sparse.linalg.spsolve(H.tocsr(), b.tocsr())
            return deltax, e
        
        #build the Hessian and measurement vector
        def build_linear_system(self, x):
            r"""
            Build the linear system

            :param x: state vector
            :type x: ndarray(N)
            :return: Hessian :math:`\mat{H}(\vec{x})` and projection error :math:`\vec{b}`
            :rtype: sparse_array(N,N), sparse_ndarray(N,1), float

            Build the block structured Hessian matrix based on current bundle
            adjustment state and the Jacobians.

            :reference:
                - Robotics, Vision & Control for Python, Section 14.3.2, F.2.4, 
                  P. Corke, Springer 2023.

            :seealso: :meth:`spy` :meth:`~Camera.CentralCamera.derivatives`
            """

            # this function is slow.  lil matrices have similar speed to dok 
            # matrices
            # H += A is slower than H = H + A

            from scipy.sparse import lil_matrix
            #allocate sparse matrices
            H = lil_matrix((self.nvarstates, self.nvarstates))
            b = lil_matrix((self.nvarstates,1))

            etotal = 0

            #loop over views
            for view in self.views:
                
                # get camera pose
                k = view.index
                X = x[k:k+6]
                
                #loop over all points viewed from this camera
                for (landmark, edge) in view.incidences():

                    k = landmark.index
                    P = x[k:k+3]  # get landmark position
            
                    # for this view and landmark, get observation
                    uv = edge.p
                    
                    # compute Jacobians and predicted projection
                    uvhat, JA, JB = self.camera.derivatives(X, P)

                    # compute reprojection error as a column vector
                    e = np.c_[uvhat - uv]
                    etotal = etotal + e.T @ e
                    
                    i = view.index2
                    j = landmark.index2

                    # compute the block components of H and b for this edge
                    if not view.isfixed and not landmark.isfixed:
                        # adjustable point and view
                        H_ii = JA.T @ JA
                        H_ij = JA.T @ JB
                        H_jj = JB.T @ JB
                        
                        H[i:i+6, i:i+6] = H[i:i+6, i:i+6] + H_ii
                        H[i:i+6, j:j+3] = H[i:i+6, j:j+3] + H_ij
                        H[j:j+3, i:i+6] = H[j:j+3, i:i+6] + H_ij.T
                        H[j:j+3, j:j+3] = H[j:j+3, j:j+3] + H_jj
                        
                        b[i:i+6, 0] = b[i:i+6, 0] - JA.T @ e
                        b[j:j+3, 0] = b[j:j+3, 0] - JB.T @ e
                        
                    elif view.isfixed and not landmark.isfixed:
                        # fixed camera and adjustable point
                        
                        H[j:j+3, j:j+3] = H[j:j+3, j:j+3] + JB.T @ JB
                        b[j:j+3, 0] = b[j:j+3, 0] - JB.T @ e
                        
                    elif not view.isfixed and landmark.isfixed:
                        # adjustable camera and fixed point
                        
                        H[i:i+6, i:i+6] = H[i:i+6, i:i+6] + JA.T @ JA
                        b[i:i+6, 0] = b[i:i+6, 0] - JA.T @ e

            return H, b, etotal
                
        def spyH(self, x, block=False):
            """
            Display sparsity of Hessian

            :param x: state vector
            :type x: ndarray(N)

            Use Matplotlib to display the zero and non-zero elements of the
            Hessian.

            :seealso: :meth:`build_linear_system` 
            """
            H, *_ = self.build_linear_system(x)
            plt.spy(H)
            plt.show(block=True)
        
        def getstate(self):
            """
            Get the state vector

            :return: state  vector
            :rtype: ndarray(N)

            Build the state vector by concatenating the pose of all cameras and
            then the position of all landmarks.  That information is provided at
            problem initialization by calls to :meth:`add_view` and
            :meth:`add_landmark`.

            :seealso: :meth:`setstate` :meth:`nstates` :meth:`add_view` :meth:`add_landmark`
            """
            x = []
            
            for view in self.views:  #step through camera nodes
                x.extend(view.coord)

            for landmark in self.landmarks:  #step through landmark nodes
                x.extend(landmark.coord)

            return np.array(x)
        
        def setstate(self, x):
            """
            Update camera and landmark state

            :param x: new state vector
            :type x: ndarray(N)

            Copy new state data into the nodes of the bundle adjustment graph.
            Those nodes corresponding to fixed cameras or landmarks are
            unchanged.

            :seealso: :meth:`updatestate` :meth:`getstate`
            """
            
            for view in self.views:  #step through view nodes
                X = x[:6]
                x = x[6:]
                if not view.isfixed:
                    view.coord = X
            
            for landmark in self.landmarks:
                    X = x[:3]
                    x = x[3:]
                    if not landmark.isfixed:
                        landmark.coord = X

        def updatestate(self, x, dx):
            """
            Update the state vector

            :param x: state vector
            :type x: ndarray(N)
            :param dx: variable state update vector
            :type dx: ndarray(M)
            :return: updated state vector
            :rtype: ndarray(N)

            The elements of the update to the variable state are inserted into
            the state vector.  Those elements corresponding to fixed cameras or
            landmarks are unchanged.

            :seealso: :meth:`setstate` :meth:`nstates`
            """
            xnew = np.zeros(x.shape)

            # for each camera we need to compound the camera pose with the
            # incremental relative pose
            for view in self.views:
                k = view.index
                if view.isfixed:
                    xnew[k:k+6] = x[k:k+6]
                else:
                    # current pose
                    X = x[k:k+6]
                    t = X[:3]
                    qv = X[3:]
                    
                    # incremental pose
                    k2 = view.index2
                    dX = dx[k2:k2+6]
                    dt = dX[:3]
                    dqv = dX[3:]

                    tnew = t + dt  #assume translation in old frame
                    qvnew = UnitQuaternion.qvmul(qv, dqv)
                    
                    xnew[k:k+6] = np.r_[tnew, qvnew]

            #for each landmark we add the increment to its position
            for landmark in self.landmarks:
                k = landmark.index
                P = x[k:k+3]
                if landmark.isfixed:
                    xnew[k:k+3] = P
                else:
                    k2 = landmark.index2
                    dP = dx[k2:k2+3]
                    xnew[k:k+3] = P + dP

            return xnew
        
        #Compute total squared reprojection error
        def errors(self, x=None):
            """
            Total reprojection error

            :param x: state vector, defaults to state vector in instance
            :type x: ndarray(N), optional
            :return: total residual
            :rtype: float

            Compute the total reprojection error, of all projected landmarks
            on all camera viewpoints. Is ideally zero.

            :seealso: :meth:`getresidual`
            """
            
            if x is None:
                x = self.getstate()
            r = self.getresidual(x)
            
            return np.sum(r)
        
        def getresidual(self, x=None):
            r"""
            Get error residuals

            :param X: state vector, defaults to state vector in instance
            :type X: ndarray(N), optional
            :return: residuals :math:`\mat{R}` for each observation
            :rtype: ndarray(V,L)

            Returns a 2D array :math:`\mat{R}` whose elements :math:`r_{ij}`
            represent the Euclidean reprojection error for camera :math:`i`
            observing landmark :math:`j`.

            :seealso: :meth:`errors`
            """
            # this is the squared reprojection errors
            self.update_index()

            if x is None:
                x = self.getstate()
            
            residual = np.zeros((self.nviews, self.nlandmarks))
            # loop over views
            for view in self.views:
                
                # get view pose
                k = view.index
                X = x[k:k+6]

                # loop over all points viewed from this camera
                for (landmark, edge) in view.incidences():

                    k = landmark.index
                    P = x[k:k+3]  # get landmark position
                    
                    uv = edge.p
                    
                    uvhat, *_ = self.camera.derivatives(X, P)
                    if np.any(np.isnan(uvhat)):
                        print('bad uvhat in residual')
                    
                    # compute reprojection error
                    e = uvhat - uv
                    residual[view.id, landmark.id] = np.dot(e, e)
            return residual

        @property
        def graph(self):
            """
            Get the scene graph

            :return: scene graph
            :rtype: :class:`PGraph`

            The scene graph has nodes representing camera viewpoints, of type
            :class:`ViewPoint`, and nodes representing landmarks, of type
            :class:`Landmark`. An edge, of type :class:`Observation`, exists
            between a landmark and the viewpoint that observed, and the edge has
            the associated image plane projection.
            """
            return self.g

        def plot(self, camera={}, ax=None, **kwargs):
            """
            Plot the scene graph

            :param camera: options passed to :obj:`CentralCamera.plot`, defaults to {}
            :type camera: dict, optional
            :param ax: axis to plot into, defaults to None
            :type ax: Axes, optional
            :param kwargs: options passed to :obj:`PGraph.plot`

            Display the nodes and edges of the scene graph as an embedded graph.
            Overlay camera icons to indicate the camera viewpoint nodes.

            :seealso: :meth:`graph`
            """
            if ax is None:
                plt.clf()
                ax = base.plotvol3()
            self.g.plot(**kwargs) #edge=dict(color=0.8*np.r_[1, 1, 1]), **kwargs)
            # ax.set_aspect('equal')
            
            # colorOrder = get(gca, 'ColorOrder')
            for view in self.views:
                cam = self.camera.move(view.pose)
                # cidx = mod(i-1, numrows(colorOrder))+1
                # color = colorOrder(cidx,:)
                cam.plot(pose=view.pose, ax=ax, color=view._color, **camera) # 'color', color, 'persist')
            # ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            plt.grid(True)

        def __repr__(self):
            """
            String representation

            :return: multiline string describing key parameters of bundle adjustment problem
            :rtype: str
            """
            return str(self)
            
        def __str__(self):
            """
            String representation

            :return: multiline string describing key parameters of bundle adjustment problem
            :rtype: str
            """
            s = 'Bundle adjustment problem:'
            s += f"  {self.nviews} views\n"
            fixedcam = [i for i, view in enumerate(self.views) if view.isfixed]
            if len(fixedcam) > 0:
                s += f"    {len(fixedcam)} locked views: {fixedcam}\n"
            fixedlandmarks = [i for i, landmark in enumerate(self.landmarks) if landmark.isfixed]
            if len(fixedlandmarks) > 0:
                s += f"    {len(fixedlandmarks)} locked landmarks: {fixedlandmarks}\n"

            s += f"  {self.nlandmarks} landmarks\n"
            
            s += f"  {self.g.ne} projections\n"
            
            s += f"  {self.nstates} total states\n"
            s += f"  {self.nvarstates} variable states\n"
            s += f"  {self.g.ne * 2} equations\n"
            v = np.array(self.g.connectivity(self.views))
            s += f"  landmarks per view: min={v.min():d}, max={v.max():d}, avg={v.mean():.1f}\n"
            l = np.array(self.g.connectivity(self.landmarks))
            s += f"  views per landmark: min={l.min():d}, max={l.max():d}, avg={l.mean():.1f}\n"
            return s
else:
    class BundleAdjust:
        pass

if __name__ == "__main__":

    from spatialmath import UnitQuaternion

    ba = BundleAdjust.load_sba('7cams.txt', '7pts.txt', 'calib.txt')
    print(ba)
    print(ba.camera)

    ba.optimize(verbose=False)