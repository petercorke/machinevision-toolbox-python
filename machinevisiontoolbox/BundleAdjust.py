import time
import numpy as np
import scipy as sp
import pgraph
from spatialmath import base
from spatialmath import SE3, SO3, UnitQuaternion
import matplotlib.pyplot as plt

from machinevisiontoolbox import CentralCamera

# We use the PGraph graph package and subclass the nodes and edges for an
# undirected graph
#
# Each node has an index into the state vector given by its index and index2
# properties.  These are initialized by a call to update_index
 
class _Common:
    @property
    def index(self):
        """
        Index into the state vector

        :return: the index of the start of this object's state in the state vector
        :rtype: int
        """
        return self._index

    @property
    def index2(self):
        """
        Index into the variable state vector

        :return: the index of the start of this object's state in the variable state vector
        :rtype: int
        """
        return self._index2

    @property
    def isfixed(self):
        return self._fixed

class ViewNode(pgraph.UVertex, _Common):

    def __init__(self, x, fixed=False):
        super().__init__()
        self.coord = x
        self._fixed = fixed

    @property
    def pose(self):
        t = self.coord[:3]
        qv = self.coord[3:]

        return SE3(t) * UnitQuaternion.Vec3(qv).SE3()

class LandmarkNode(pgraph.UVertex, _Common):
    def __init__(self, P, fixed=False):
        super().__init__()
        self.P = P
        self._fixed = fixed

class Observation(pgraph.Edge):
    def __init__(self, camera, landmark, uv):
        super().__init__(camera, landmark, cost=0)
        self.uv = uv

class BundleAdjust:
    """
    Implementation of a workable, easy to follow, but simplistic, bundle
    adjustment algorithm.

    It uses SciPy sparse linear algebra functions to solve the update equation.

    The state vector comprises, in order:

    - a 6-vector for every view, the camera pose (tx, ty, tz, qx, qy, qz) as 
      translation and vector part of unit quaternion
    - a 3-vector for every landmark, (x, y, Z)

    Cameras and landmarks can be fixed in which case we have a variable state
    vector, shorter than the state vector, holding only the states corresponding
    to movable cameras and landmarks.
    """

    def __init__(self, camera):
        """
        Setup a bundle adjustment problem

        :param camera: model of the moving camera
        :type camera: CentralCamera instance

        .. warning:: This class assumes that all camera views have the same camera
            intrinsics.

        :seealso: :class:`~machinevisiontoolbox.CentralCamera`
        """
        self.camera = camera

        # we use a PGraph object to represent nodes and edges as an undirected graph:
        #  - ViewNode class for vertices representing camera poses as (tx, ty,
        #    tz, qx, qy, qz)
        #  - LandmarkNode for  vertices representing landmark positions as (x,
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

:       seealso: :meth:`.add_view`
        """
        return self._nviews

    @property
    def nlandmarks(self):
        """
        Number of landmarks

        :return: Number of landmarks
        :rtype: int

        :seealso: :meth:`.add_landmark`
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

        :seealso: :meth:`.nvarstate`
        """
        return  6 * self.nviews + 3 * self.nlandmarks

    @property
    def nvarstates(self):
        """
        Length of variable state vector

        :return: Length of the variable state vector
        :rtype: int

        This excludes fixed views and landmarks, only includes cameras
        and landmarks whose state will be updated in the opimization.

        :seealso: :meth:`.nstates`
        """
        return  6 * (self.nviews - len(self.fixedviews)) \
              + 3 * (self.nlandmarks - len(self.fixedlandmarks))

    def add_view(self, pose=None, fixed=False):
        """
        Add camera view to bundle adjustment problem

        :param pose: camera pose, defaults to None
        :type pose: SE3 or array_like(7)
        :param fixed: the camera is fixed, defaults to False
        :type fixed: bool, optional
        :return: [description]
        :rtype: ViewNode instance
        
        Creates a camera node and adds it to the bundle adjustment.
        
        The camera ``pose``  can be ``SE3`` or a vector (1x7) comprising
        translation and unit quaternion in vector form.

        If the camera is fixed (anchored) it will not be adjusted in the
        optimization process.
        
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

        v = ViewNode(x, fixed=fixed)
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

        :param P: 3D landmark point
        :type P: array_like(3)
        :param fixed: the landmark is fixed, defaults to False
        :type fixed: bool, optional
        :return: [description]
        :rtype: LandmarkNode instance

        A landmark node added to the bundle adjustment problem.  The landmark has position P (3x1).

        If the landmark is fixed (anchored) it will not be adjusted in the
        optimization process.
        
        :seealso: :meth:`add_view` :meth:`add_projection` 
        """
        base.assertvector(P, 3)
        
        # P = np.r_[P, 0, 0, 0]

        l = LandmarkNode(P, fixed=fixed)
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

    def add_projection(self, view, landmark, uv):
        """
        Add camera observation to bundle adjustment problem

        :param view: [description]
        :type view: ViewNode instance
        :param landmark: [description]
        :type landmark: LandmarkNode instance
        :param uv: image plane coordinate
        :type uv: array_like(2)
        
        Add observation by ``view`` of a ``landmark`` to the bundle adjustment problem.  
        This is an edge connecting a camera node to a landmark node.

        :seealso: :meth:`add_view` :meth:`add_landmark` :class:`PGraph`
        """
        assert len(uv) == 2, 'uv must be a 2-vector'
        
        edge = Observation(view, landmark, uv.flatten()) # create edge object
        e = view.connect(landmark, edge=edge)  # connect nodes with it
        e.name = view.name + "--" + landmark.name

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
        
            ba = Bundle.load_SBA('7cams.txt', '7pts.txt', 'calib.txt')
            X = ba.optimize()

        :reference: Sparse Bundle Adjustment package by Manolis Lourakis,
            http://users.ics.forth.gr/~lourakis/sba
        
        :seealso: :meth:`.add_view` :meth:`.add_landmark`, BundleAdjust.add_projection, PGraph.
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
    
    def optimize(self, X=None, animate=False, lmbda=0.1, 
        lmbdamin=1e-8, dxmin=1e-4, tol=0.5, iterations=1000, verbose=False):
        """
        Perform the bundle adjustment

        :param X: state vector, if None use the state vector in the object
        :type X: ndarray(N), optional
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
            falls below this threshold, defaults to 0.5
        :type tol: float, optional
        :param iterations: maximum number of iterations, defaults to 1000
        :type iterations: int, optional
        :param verbose: show Levenberg-Marquadt status, defaults to False
        :type verbose: bool, optional
        :return: optimized state vector
        :rtype: ndarray(N)

        Performs a Levenberg-Marquadt style optimization of the bundle
        adjustment problem.  Adjusts camera poses and landmark positions in
        order to minimize the total reprojection error.
        """

        self.update_index()
        
        if X is None:
            X = self.getstate()
        X0 = X
        
        t0 = time.perf_counter()
        
        for i in range(iterations):
            if animate:
                if not retain:
                    plt.clf()
                g2.plot()
                plt.pause(0.5)
            
            ta = time.perf_counter()
            # solve for the step
            dX, energy = self.solve(X, lmbda)

            # update the state
            Xnew = self.updatestate(X, dX)

            # compute new value of cost
            enew = self.errors(Xnew)
            
            dt = time.perf_counter() - ta
            print(f"Bundle adjustment cost {enew:.3g} (solved in {dt:.2f} sec)")
            print(energy)
            # are we there yet?
            if energy < tol:
                break
            
            # have we stopped moving
            if  base.norm(dX) < dxmin:
                break

            # do the Levenberg-Marquadt thing, was it a good update?
            if enew < energy:
                # step is accepted
                X = Xnew
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
        print(f"\n * {i + 1} iterations in {tf - t0:.1f} seconds")
        print(f" * Final RMS error is {np.sqrt(enew / self.g.ne):.2f} pixels")
        
        return Xnew
    
    def solve(self, X, lmbda=0.0):
        """
        Solve for state update

        :param X: state vector
        :type X: ndarray(N)
        :param lmbda: damping term, defaults to 0.0
        :type lmbda: float, optional
        :return: dX, update to the variable state vector
        :rtype: ndarray(M)

        Determines the state update dX by creating and solving the linear
        equation :math:`\mat{H} \mat{\Delta_X} = \mat{b}` where :math:`\mat{H}` is the
        Hessian and :math:`\mat{b}` is the ???

        .. note:: The damping term ``lmbda`` is added to the diagonal of the Hessian to
            prevent problems when the Hessian is nearly singular.

        .. note:: If cameras or landmarks are fixed then ``len(dX) < len(X)``
            since fixed elements are omitted from the variable state vector
            used for the optimization.

        :seealso: :meth:`.build_linear_system`
        """
        # create the Hessian and error vector
        H, b, e = self.build_linear_system(X)
        
        # add damping term to the diagonal
        for i in range(self.nvarstates):
            H[i, i] += lmbda
        
        # solve for the state update
        #- could replace this with the Schur complement trick
        deltax = sp.sparse.linalg.spsolve(H.tocsr(), b.tocsr())
        return deltax, e
    
    #build the Hessian and measurement vector
    def build_linear_system(self, X):
        """
        Build the linear system

        :param X: state vector
        :type X: ndarray(N)
        :return: Hessian, projection error
        :rtype: sparse_array(N,N), sparse_ndarray(N,1), float

        Build the block structured Hessian matrix based on current bundle
        adjustment state and the Jacobians.

        :reference: 

        :seealso: :meth:`~Camera.CentralCamera.derivatives`
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
            x = X[k:k+6]
            
            #loop over all points viewed from this camera
            for (landmark, edge) in view.incidences():

                k = landmark.index
                P = X[k:k+3]  # get landmark position
        
                # for this view and landmark, get observation
                uv = edge.uv
                
                # compute Jacobians and predicted projection
                uvhat, JA, JB = self.camera.derivatives(x, P)

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
            
    def spyH(self, X):
        H, *_ = self.build_linear_system(X)
        plt.spy(H)
        plt.pause(2)
    
    def getstate(self):
        """
        Get the state vector

        :return: state  vector
        :rtype: ndarray(N)

        Build the state vector by concatenating the pose of all cameras and
        then the position of all landmarks.  That information is provided
        at problem initialization by calls to ``add_view()`` and ``add_landmark()``.

        :seealso: :meth:`add_view` :meth:`add_landmark`
        """
        X = []
        
        for view in self.views:  #step through camera nodes
            X.extend(view.coord)

        for landmark in self.landmarks:  #step through landmark nodes
            X.extend(landmark.coord)

        return np.array(X)
    
    def setstate(self, X):
        """
        Update camera and landmark state

        :param X: state vector
        :type X: ndarray(N)

        Copy state data into the nodes of the bundle adjustment graph.
        """
        
        for view in self.views:  #step through view nodes
            x = X[:6]
            X = X[6:]
            if not view.isfixed:
                view.coord = x
        
        for landmark in self.landmarks:
                x = X[:3]
                X = X[3:]
                if not landmark.isfixed:
                    landmark.coord = x

    def updatestate(self, X, dX):
        """
        Update the state vector

        :param X: state vector
        :type X: ndarray(N)
        :param dX: variable state update vector
        :type dX: ndarray(M)
        :return: updated state vector
        :rtype: ndarray(N)

        The elements of the update to the variable state are inserted into
        the state vector.  Those elements corresponding to fixed cameras or
        landmarks are unchanged.
        """
        Xnew = np.zeros(X.shape)

        # for each camera we need to compound the camera pose with the
        # incremental relative pose
        for view in self.views:
            k = view.index
            if view.isfixed:
                Xnew[k:k+6] = X[k:k+6]
            else:
                # current pose
                x = X[k:k+6]
                t = x[:3]
                qv = x[3:]
                
                # incremental pose
                k2 = view.index2
                dx = dX[k2:k2+6]
                dt = dx[:3]
                dqv = dx[3:]

                tnew = t + dt  #assume translation in old frame
                qvnew = UnitQuaternion.qvmul(qv, dqv)
                
                Xnew[k:k+6] = np.r_[tnew, qvnew]

        #for each landmark we add the increment to its position
        for landmark in self.landmarks:
            k = landmark.index
            P = X[k:k+3]
            if landmark.isfixed:
                Xnew[k:k+3] = P
            else:
                k2 = landmark.index2
                dP = dX[k2:k2+3]
                Xnew[k:k+3] = P + dP

        return Xnew
    
    #Compute total squared reprojection error
    def errors(self, X=None):
        
        if X is None:
            X = self.getstate()
        r = self.getresidual(X)
        
        return np.sum(r)
    
    def getresidual(self, X=None):
        # this is the squared reprojection errors
        self.update_index()

        if X is None:
            X = self.getstate()
        
        residual = np.zeros((self.nviews, self.nlandmarks))
        # loop over views
        for view in self.views:
            
            # get view pose
            k = view.index
            x = X[k:k+6]

            # loop over all points viewed from this camera
            for (landmark, edge) in view.incidences():

                k = landmark.index
                P = X[k:k+3]  # get landmark position
                
                uv = edge.uv
                
                uvhat, *_ = self.camera.derivatives(x, P)
                if np.any(np.isnan(uvhat)):
                    print('bad uvhat in residual')
                
                # compute reprojection error
                e = uvhat - uv
                residual[view.id, view.id] = np.dot(e, e)
        return residual

    def plot(self, **kwargs):
        plt.clf()
        self.g.plot() #edge=dict(color=0.8*np.r_[1, 1, 1]), **kwargs)
        # ax.set_aspect('equal')
        
        # colorOrder = get(gca, 'ColorOrder')
        for view in self.views:
            cam = self.camera.move(view.pose)
            # cidx = mod(i-1, numrows(colorOrder))+1
            # color = colorOrder(cidx,:)
            cam.plot_camera(pose=view.pose, scale=0.2) # 'color', color, 'persist')
        # ax.set_aspect('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.gca().set_zlabel('Z (m)')
        plt.grid(True)

    def __str__(self):
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
        v = np.array(self.g.connectivity(self.views))
        s += f"  landmarks per view: min={v.min():d}, max={v.max():d}, avg={v.mean():.1f}\n"
        l = np.array(self.g.connectivity(self.landmarks))
        s += f"  views per landmark: min={l.min():d}, max={l.max():d}, avg={l.mean():.1f}\n"
        return s


if __name__ == "__main__":

    from spatialmath import UnitQuaternion

    ba = BundleAdjust.load_sba('7cams.txt', '7pts.txt', 'calib.txt')
    print(ba)
    print(ba.camera)

    ba.optimize(verbose=False)