#!/usr/bin/env python3
from abc import ABC
# from machinevisiontoolbox.Camera import P
import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox.base import mkgrid
from machinevisiontoolbox import CentralCamera
import spatialmath.base as smbase
from spatialmath import SE3

class VisualServo(ABC):

    class _history:
        pass

    def __init__(self, camera, niter=100, graphics=True, fps=5, pose_g=None, 
            pose_0=None, pose_d=None, P=None, p_d=None, 
            title=None, plotvol=None, movie=None, type=None, verbose=False):
        """
        Visual servo abstract superclass

        :param camera: camera model
        :type camera: Camera subclass
        :param niter: number of simulation iterations, can be overridden with ``run``, defaults to 100
        :type niter: int, optional
        :param graphics: show graphical animation, defaults to True
        :type graphics: bool, optional
        :param fps: graphics display frames per second, defaults to 5
        :type fps: float, optional
        :param pose_g: pose of goal frame {G}, for PBVS  only, defaults to None
        :type pose_g: SE3 instance, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param pose_d: desired pose of goal {G} with respect to camera, defaults to None
        :type pose_d: SE3 instance, optional
        :param P: world points, defaults to None
        :type P: array_like(3,N), optional
        :param p_d: desired image plane points, defaults to None
        :type p_d: array_like(2,N), optional
        :param plotvol: [description], defaults to None
        :type plotvol: [type], optional
        :param movie: [description], defaults to None
        :type movie: [type], optional
        :param type: [description], defaults to None
        :type type: [type], optional
        :param verbose: rint out extra information during simulation, defaults to False
        :type verbose: bool, optional

        Two windows are shown and animated:
        - The camera view, showing the desired view (*) and the 
          current view (o)
        - The external view, showing the target points and the camera

        .. warning:: The pose of the camera object is modified while the simulation runs. 

        """

        self.camera = camera
        self.history = []

        self.niter = niter

        self.graphics = graphics
        self.fps = fps
        self.verbose = verbose
        self.pose_d = pose_d
        self.pose_g = pose_g

        self.pose_0 = pose_0
            
        self.P = P
        if P is not None:
            self.npoints = P.shape[1]
        if p_d is None:
            p_d = camera.project_point(P, pose=pose_d)
        self.p_star = p_d

        # self.ax = smbase.plotvol3(plotvol)
        self.movie = movie
        self.type = type
        if graphics:

            fig = plt.figure(figsize=plt.figaspect(0.5))
            self.fig = fig

            # First subplot
            ax = fig.add_subplot(1, 2, 1)
            self.camera._init_imageplane(ax=ax)
            self.ax_camera = ax

            # Second subplot
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax = smbase.plotvol3(plotvol, ax=ax)
            # smbase.plot_sphere(0.06, self.P, color='r', ax=ax)
            # self.camera.plot(self.pose, label=True, ax=ax)
            ax.view_init(16, 28)
            plt.grid(True)
            self.ax_3dview = ax

            self.camera_args = dict(shape='camera', color='b', scale=0.3, ax=self.ax_3dview)

            fig.patch.set_color('#f0f0f0')
            self.ax_3dview.set_facecolor('#f0f0f0')

            if title is not None:
                self.fig.canvas.manager.set_window_title(title)

    def init(self):
        """
        Initialize visual servo simulation.

        This is common initialization used by derived classes.  It initializes
        the image plane and world view graphics.

        If ``pose_0`` was specified at constructor time, the camera is set
        to that pose.  Otherwise the camera is ``reset`` which returns it to 
        the pose it has when constructed.

        :meth:`Camera.reset`
        """

        # initialize the camera pose
        if self.pose_0 is not None:
            self.camera.pose = self.pose_0
        else:
            self.camera.reset()

        # do graphics setup
        if self.graphics:

            # show the initial image plane projections as circles
            self.camera.clf()
            self.camera.plot_point(self.P, objpose=self.pose_g, markersize=8, markerfacecolor='none')

            # show the points in the world view
            if self.pose_g is not None:
                P = self.pose_g * self.P
            else:
                P = self.P
            smbase.plot_sphere(0.05, P, color='r', ax=self.ax_3dview)
            self.camera.plot(**{**self.camera_args, **dict(color='k')})

        # clear the history
        self.history = []

    def run(self, niter=None):
        """
        Run visual servo simulation

        :param niter: number of simulation iterations, defaults to value given
            to constructor
        :type niter: int, optional

        Concrete method that invokes the ``step`` method of the derived
        class which returns
        a flag to indicate if the simulation is complete.
        """
        self.init()
        
        if self.movie is not None:
            self.anim = Animate(self.movie)
        
        if niter is None:
            niter = self.niter

        alpha_min = 0.1
        for step in range(niter):
            
            status = self.step(step)

            if self.graphics:
                if self.type == 'point':
                    self.plot_point(self.history[-1].p, markersize=4)

                self.clear_3dview()
                alpha = alpha_min + (1 - alpha_min) * step / niter
                self.camera.plot(alpha=alpha, **self.camera_args)

                if self.movie is not None:
                    self.anim.add()
                else:
                    plt.pause(1 / self.fps)
            
            if status > 0:
                print('completed on error tolerance')
                break
            elif status < 0:
                print('failed on error\n')
                break

        else:
            # exit on iteration limit
            print('completed on iteration count')
        
        if self.movie is not None:
            self.anim.close()

    def plot_p(self):
        """
        Plot feature trajectory from simulation

        Show image feature points versus time.

        :seealso: :meth:`plot_vel` :meth:`self.plot_pose` :meth:`plot_jcond` :meth:`plot_z` :meth:`plot_error`
        """
        
        if len(self.history) == 0:
            return

        if self.type != 'point':
            print('Can only plot image plane trajectories for point-based IBVS')
            return

        # result is a vector with row per time step, each row is u1, v1, u2, v2 ...
        for i in range(self.npoints):
            u = [h.p[0, i] for h in self.history]  # get data for i'th point
            v = [h.p[1, i] for h in self.history]
            plt.plot(u, v, 'b')
        
        # mark the initial target shape
        smbase.plot_polygon(self.history[0].p, 'o--', close=True, markeredgecolor='k', markerfacecolor='w', label='initial')
        
        # mark the goal target shape
        if isinstance(self, IBVS):
            smbase.plot_polygon(self.p_star, 'k*:', close=True, markeredgecolor='k', markerfacecolor='k', label='goal')

        if isinstance(self, PBVS):
            p = self.camera.project_point(self.P, pose=self.pose_d.inv())
            smbase.plot_polygon(p, 'k*:', close=True, markeredgecolor='k', markerfacecolor='k', label='goal')
            

        # axis([0 self.camera.npix[0] 0 self.camera.npix[1]])
        # daspect([1 1 1])
        plt.grid(True)
        plt.xlabel('u (pixels)')
        plt.ylabel('v (pixels)')
        plt.xlim(0, self.camera.width)
        plt.ylim(0, self.camera.height)
        plt.legend()
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')  
        ax.set_facecolor('lightyellow')
    

    def plot_vel(self):
        """
        Plot camera velocity from simulation

        Show camera velocity versus time.

        :seealso: :meth:`plot_p` :meth:`self.plot_pose` ::meth:`plot_jcond` :meth:`plot_z` :meth:`plot_error`
        """
        if len(self.history) == 0:
            return

        vel = np.array([h.vel for h in self.history])
        plt.plot(vel[:, :3], '-')
        plt.plot(vel[:, 3:], '--')
        plt.ylabel('Cartesian velocity')
        plt.grid(True)
        plt.xlabel('Time step')
        plt.xlim(0, len(self.history) - 1)
        plt.legend(['$v_x$', '$v_y$', '$v_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'], loc='upper right')

    def plot_pose(self):
        """
        Plot camera trajectory from simulation

        Show camera pose versus time, as two plots: translation and rotation
        in RPY angles.

        :seealso: :meth:`plot_p` :meth:`self.plot_vel` :meth:`plot_jcond` :meth:`plot_z` :meth:`plot_error`
        """

        if len(self.history) == 0:
            return

        # Cartesian camera position vs timestep
        T = SE3([h.pose for h in self.history])
        
        plt.subplot(211)
        plt.plot(T.t)
        plt.xlim(0, len(self.history) - 1)
        plt.ylabel('Camera position (m)')
        plt.legend(['x', 'y', 'z'])
        plt.grid(True)
        
        plt.subplot(212)
        plt.plot(T.rpy(order='camera'))
        plt.ylabel('Camera orientation (rad)')
        plt.grid(True)
        plt.xlabel('Time step')
        plt.xlim(0, len(self.history) - 1)
        plt.legend([r'$\alpha$', r'$\beta$', r'$\gamma$'])


    def plot_jcond(self):
        """
        Plot image Jacobian condition from simulation.

        Show image Jacobian condition versus time. Indicates whether the point configuration is close to
        singular.

        :seealso: :meth:`plot_p` :meth:`self.plot_vel` :meth:`self.plot_pose` :meth:`plot_z` :meth:`plot_error`
        """
        
        if len(self.history) == 0:
            return
        
        Jcond = [h.jcond for h in self.history]
        # Image Jacobian condition number vs time
        plt.plot(Jcond)
        plt.grid(True)
        plt.ylabel('Jacobian condition number')
        plt.xlabel('Time step')
        plt.xlim(0, len(self.history) - 1)

    def plot_z(self):
        """
        Plot feature depth from simulation

        Show depth of all features versus time. If a depth estimator is
        used it shows true and estimated depth.

        :seealso: :meth:`plot_p` :meth:`self.plot_vel`  :meth:`self.plot_pose` :meth:`plot_jcond` :meth:`plot_error`
        """
        if len(self.history) == 0:
            return
            
        if self.type != 'point':
            print('Z-estimator data only computed for point-based IBVS')
            return

        Z_est = np.array([h.Z_est for h in self.history])
        Z_true = np.array([h.Z_true for h in self.history])
        plt.plot(Z_true, '-', label='true')
        plt.plot(Z_est, '--', label='estimate')
        plt.grid()
        plt.ylabel('Depth (m)')
        plt.xlabel('Time step')
        plt.xlim(0, len(self.history) - 1)
        plt.legend()

    def plot_error(self):
        """
        Plot feature error from simulation

        Show error of all features, norm of (desired - actual) versus time. If a depth estimator is
        used it shows true and estimated depth.

        :seealso: :meth:`plot_p` :meth:`self.plot_vel`  :meth:`self.plot_pose` :meth:`plot_jcond` :meth:`plot_z`
        """
        
        if len(self.history) == 0:
            return
        
        e = np.array([h.e for h in self.history])
        if self.type == 'point':
            plt.plot(e[:, 0::2], 'r')
            plt.plot(e[:, 1::2], 'b')
            plt.ylabel('Feature error (pixel)')
            
            plt.legend('u', 'v')
        else:
            plot(e)
            plt.ylabel('Feature error')

        plt.grid(True)
        plt.xlabel('Time')
        plt.xlim(0, len(self.history))

        return e

    def plot_all(self):
        """
        Plot all data from simulation

        Show simulation results, in separate figures, feature values, velocity, 
        error and camera pose versus time.

        :seealso: :meth:`plot_p` :meth:`self.plot_vel`  :meth:`self.plot_pose` :meth:`plot_jcond` :meth:`plot_z` :meth:`plot_error`
        """

        plt.figure()
        self.plot_p()

        plt.figure()
        self.plot_vel()

        plt.figure()
        self.plotpose()

        plt.figure()
        self.plot_error()

        # optional plots depending on what history was recorded
        if hasattr(history[0], 'Z_est'):
            plt.figure()
            self.plot_z()
        
        if hasattr(self.history[0], 'jcond'):
            plt.figure()
            self.plot_jcond()

    def __str__(self):
        """
        String representation of visual servo object

        :return: compact string representation
        :rtype: str
        """
        s = f"Visual servo object: camera={self.camera.name}\n  {self.niter} iterations, {len(self.history)} history'"

        s += np.array2string(self.P, prefix='P = ')
        if self.pose_0 is not None:
            s +=  "\n" + self.pose_0.strline(label='cT', orient='camera')
        if self.pose_d is not None:
            s +=  "\n" + self.pose_d.strline(label='cdTg', orient='camera')
        return s

    def __repr__(self):
        return str(self)

    def plot_point(self, *args, **kwargs):
        return self.camera.plot_point(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.camera.plot(*args, ax=self.ax_3dview, **kwargs)

    def clear_3dview(self):
        for child in self.ax_3dview.get_children(): # ax.lines:
            if __class__.__name__ == 'Line3DCollection':
                child.remove()


class PBVS(VisualServo):

    def __init__(self, camera, eterm=0, lmbda=0.05, **kwargs):
        """
        Position-based visual servo class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param eterm: termination threshold on residual error, defaults to 0
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.05
        :type lmbda: float, optional
        :param P: world points in frame {G}, defaults to None
        :type P: array_like(3,N), optional
        :param pose_g: pose of goal frame {G}, for PBVS  only, defaults to None
        :type pose_g: SE3 instance, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional

        Example::

            cam = CentralCamera('default');
            Tc0 = transl(1,1,-3)*trotz(0.6);
            TcStar_t = transl(0, 0, 1);
            pbvs = PBVS(cam, 'T0', Tc0, 'Tf', TcStar_t);
            pbself.plot_p

        References::
        - Robotics, Vision & Control, Chap 15
        P. Corke, Springer 2011.

        .. note:: The history attribute is a vector of structures each of which is a snapshot at
            each simulation step of information about the image plane, camera pose, error, 
            Jacobian condition number, error norm, image plane size and desired feature 
            locations.

        :seealso: :class:`IBVS` :class:`IBVS_l` :class:`IBVS_e` :class:`IBVS_polar` :class:`IBVS_sph`
        """
        
        # invoke superclass constructor
        super().__init__(camera, type='point', title='PBVS simulation', **kwargs)

        self.eterm = eterm
        self.lmbda = lmbda

        if self.pose_d is None:
            self.pose_d = SE3(0, 0, 1)
            print('setting Tf to default')

    def step(self, t):
        """
        Compute one timestep of PBVS simulation.

        :param t: simulation time
        :type t: float
        :return: simulation status, 0 if OK, 1 if terminating
        :rtype: int

        Called by the ``run`` method and performs the following steps:

        * find projections in current camera view
        * using world point data, estimate goal pose {G} with respect to camera pose 
        * incrementally update the camera pose.

        :seealso: :meth:`run` :meth:`VisualServo.run`
        """
        
        status = 0;

        # compute the current view
        uv = self.camera.project_point(self.P, objpose=self.pose_g)

        # estimate pose of goal wrt camera
        Te_C_G = self.camera.estpose(self.P, uv, frame="camera")

        # estimate motion to desired relative pose
        T_delta =  Te_C_G * self.pose_d.inv()
        
        # update the camera pose
        Td = T_delta.interp1(self.lmbda)

        self.camera.pose @= Td      # apply it to current pose

        # update the history variables
        hist = self._history()
        hist.p = uv
        vel = Td.delta()
        hist.vel = vel
        hist.pose = self.camera.pose

        self.history.append(hist)
        
        if np.linalg.norm(vel) < self.eterm:
            status = 1

        return status

class IBVS(VisualServo):

    def __init__(self, camera, eterm=0.5, lmbda=0.08, depth=None, depthest=False, vmax=None, smoothstart=None, **kwargs):
        r"""
        Image-based visual servo class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param P: world points in frame {G}, defaults to None
        :type P: array_like(3,N), optional
        :param pd: desired image plane points, defaults to None
        :type pd: array_like(2,N), optional
        :param eterm: termination threshold on residual error, defaults to 0.5
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.08
        :type lmbda: float, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param depth: depth of points, defaults to None
        :type depth: float or array_like(N), optional
        :param depthest: run simple depth estimator, defaults to False
        :type depthest: bool, optional
        :param vmax: maximum velocity, defaults to None
        :type vmax: float, optional
        :param smoothstart: enable smooth start with this value as :math:`\mu`, defaults to None
        :type smoothstart: float, optional

        Example::

                camera = CentralCamera.Default()
                Tc = trnorm( Tc * delta2tr(v) )
                Tc0 = transl(1,1,-3)*trotz(0.6)
                p_f = bsxfun(@plus, 200*[-1 -1 1 1 -1 1 1 -1], cam.pp')
                ibvs = IBVS(cam, 'T0', Tc0, 'p_f', p_f)
                self.run[]
                self.plot_p[]

        If point depth is a scalar, it applies to all points.  If an array, the
        elements are the depth for the corresponding world points.

        References::
            - Robotics, Vision & Control, Chap 15
              P. Corke, Springer 2011.

        .. note::
            - The history property is a vector of structures each of which is a
              snapshot at each simulation step of information about the image
              plane, camera pose, error, Jacobian condition number, error norm,
              image plane size and desired feature locations.
            - This implementation has a sign change compared to the task
              function notation (Chaumette papers), the the error in this code is
              desired-actual which means the control gain is positive.
        """

        # invoke superclass constructor
        super().__init__(camera, type='point', title='IBVS simulation', **kwargs)

        self.lmbda = lmbda
        self.eterm = eterm
        self.theta = 0
        self.smoothing = 0.80
        self.depth = depth
        self.depthest = depthest
        self.vmax = vmax
        self.smoothstart = smoothstart
        
    @classmethod
    def Example(cls, camera):
        print('Canned example: point-based IBVS with four feature points')
        if camera is None:
            camera = CentralCamera.Default(name='')

        P = mkgrid(2, 0.5, pose=SE3[-1,-1,2])
        pose_0 = SE3(1, 1, -3) * SE3.Rz(0.6)
        pose_d = SE3(0, 0, 1)
        self = cls(camera, P=P, pose_0=pose_0, pose_d=pose_d, depth=3)

        return self

    def init(self):
        """
        Initialize IBVS simulation.

        Implicitly called by ``run`` to initialize state variables.

        :seealso: :meth:`run` :meth:`VisualServo.init`
        """
        # initialize the vservo variables
        super().init()

        self.vel_prev = None
        self.uv_prev = None
        self.e0 = None


    def step(self, t):
        """
        Compute one timestep of IBVS simulation.

        :param t: simulation time
        :type t: float
        :return: simulation status, 0 if OK, 1 if terminating
        :rtype: int

        Called by the ``run`` method and performs the following steps:

        * find projections of world points in current camera view
        * optionally estimate point depth
        * compute the image Jacobian and camera velocity 
        * incrementally update the camera pose.

        :seealso: :meth:`run` :meth:`VisualServo.run`
        """
        
        status = 0
        Z_est = None
        
        uv = self.camera.project_point(self.P)

        hist = self._history()

        # optionally estimate depth
        if self.depthest:
            # run the depth estimator
            Z_est, Z_true = self.depth_estimator(uv)
            if self.verbose:
                print(f"Z: est={Z_est}, true={Z_true}")
            self.depth = Z_est
            if Z_est is None:
                hist.Z_est = np.zeros((self.P.shape[1],))
            else:
                hist.Z_est = Z_est.ravel()
            if Z_true is None:
                hist.Z_true = Z_true
            else:
                hist.Z_true = Z_true.ravel()

        # compute the Jacobian
        if self.depth is None:
            # exact depth from simulation (not possible in practice)
            pt = self.camera.pose.inv() * self.P
            J = self.camera.visjac_p(uv, pt[2, :])
        elif Z_est is not None:
            # use the estimated depth
            J = self.camera.visjac_p(uv, Z_est)
        else:
            # use the default depth
            J = self.camera.visjac_p(uv, self.depth)

        # compute image plane error as a column
        e = uv - self.p_star  # feature error
        e = e.flatten(order='F')  # convert columnwise to a 1D vector

        if np.linalg.norm(e) < self.eterm:
            status = 1

        # do the smoothstart trick
        #  N. Mansard and F. Chaumette, 
        #  "Task Sequencing for High-Level Sensor-Based Control," 
        #  in IEEE Transactions on Robotics, vol. 23, no. 1, pp. 60-72, Feb. 2007,
        #  doi: 10.1109/TRO.2006.889487.
        if self.smoothstart is not None:
            if self.e0 is None:
                self.e0 = e
            e -= self.e0 * np.exp(-self.smoothstart * t)

        # compute the velocity of camera in camera frame
        try:
            v = -self.lmbda * np.linalg.pinv(J) @ e
        except np.linalg.LinAlgError:
            return -1

        # limit the norm of the velocity command
        #  probably should be a weighted norm
        if self.vmax is not None:
            if np.linalg.norm(v) > self.vmax:
                v = smbase.unitvec(v) * self.vmax

        if self.verbose:
            print(v)

        # update the camera pose
        Td = SE3.Delta(v) # differential motion
        # Td = SE3(trnorm(delta2tr(v)))    
        #Td = expm( skewa(v) )
        #Td = SE3( delta2tr(v) )
        self.camera.pose @= Td       # apply it to current pose

        # update the history variables
        hist.p = uv
        vel = Td.delta()
        hist.vel = vel
        hist.e = e
        hist.enorm = np.linalg.norm(e)
        hist.jcond = np.linalg.cond(J)
        hist.pose = self.camera.pose

        self.history.append(hist)

        #TODO not really needed, its in the history
        self.vel_prev = vel
        self.uv_prev = uv

        return status

    def depth_estimator(self, uv):
        """
        Estimate depth of points.

        :param uv: current image plane points
        :type uv: array_like(2,N)
        :return: estimated and true depth of world points
        :rtype: array_like(N), array_like(N)

        Estimate point depth using a recursive least-squares update based on
        optical flow and camera incremental motion over two frames.

        """
        #TODO:
        # should have some way to initialize depth rather than assuming zero 
        # should keep Z_est as an instance variable
        
        # test if first frame
        if self.uv_prev is None:
            Z_est = None

        else:
            # compute Jacobian for unit depth, z=1
            J = self.camera.visjac_p(uv, 1)
            Jv = J[:, :3]  # velocity part, depends on 1/z
            Jw = J[:, 3:]  # rotational part, indepedent of 1/z

            # estimate image plane velocity
            uv_d =  uv.flatten(order='F') - self.uv_prev.flatten(order='F')
            
            # estimate coefficients for A (1/z) = b
            b = uv_d - Jw @ self.vel_prev[3:]
            A = Jv @ self.vel_prev[:3]

            AA = np.zeros((A.shape[0], A.shape[0]//2))
            for i in range(A.shape[0]//2):
                AA[2*i:(i+1)*2, i] = A[2*i:(i+1)*2]

            eta, resid, *_ = np.linalg.lstsq(AA, b.ravel(), rcond=None)         # least squares solution
            # eta2 = A(1:2) \ B(1:2)

            # first order smoothing
            self.theta = (1 - self.smoothing) * 1 / eta + self.smoothing * self.theta
            Z_est = self.theta

        # true depth
        P_CT = self.camera.pose.inv() * self.P
        Z_true = P_CT[2, :]

        if self.verbose:
            print('depth', Z_true)
            print('est depth', Z_est)

        return Z_est, Z_true

class IBVS_l(VisualServo):

    def __init__(self, camera, eterm=0.01, plane=None, lmbda=0.08, **kwargs):
        r"""
        Image-based visual servo for line features class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param P: world points in frame {G} define lines, defaults to None
        :type P: array_like(3,N), optional
        :param pose_d: desired pose of goal {G} with respect to camera, defaults to None
        :type pose_d: SE3 instance, optional
        :param eterm: termination threshold on residual error, defaults to 0.5
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.08
        :type lmbda: float, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param plane: plane parameters :math:`ax+by+cz+d=0`, defaults to None
        :type plane: array_like(4), optional
        :param depthest: run simple depth estimator, defaults to False
        :type depthest: bool, optional
        :param vmax: maximum velocity, defaults to None
        :type vmax: float, optional
        :param smoothstart: enable smooth start with this value as :math:`\mu`, defaults to None
        :type smoothstart: float, optional

        The world lines are defined by consecutive pairs of points in ``P``.

        The goal is defined by the image plane lines as viewed from the pose
        ``pose_d``.  Note that ``pose_d`` is not used by the controller, only
        to obtain the desired image plane lines.

        Example::

                camera = CentralCamera.Default()
                Tc = trnorm( Tc * delta2tr(v) )
                Tc0 = transl(1,1,-3)*trotz(0.6)
                p_f = bsxfun(@plus, 200*[-1 -1 1 1 -1 1 1 -1], cam.pp')
                ibvs = IBVS(cam, 'T0', Tc0, 'p_f', p_f)
                self.run[]
                self.plot_p[]

        The plane applies to all lines.

        References::
            - Robotics, Vision & Control, Chap 15
              P. Corke, Springer 2011.

        .. note::
            - The history property is a vector of structures each of which is a
              snapshot at each simulation step of information about the image
              plane, camera pose, error, Jacobian condition number, error norm,
              image plane size and desired feature locations.
            - This implementation has a sign change compared to the task
              function notation (Chaumette papers), the the error in this code is
              desired-actual which means the control gain is positive.
        """

        # invoke superclass constructor
        super().__init__(camera, type='line', **kwargs)
        
        self.eterm = eterm
        self.plane = plane
        self.lmbda = lmbda
                
    @classmethod
    def Example(cls, camera):
        # setup for a canned example
        print('Canned example: line-based IBVS with three lines')
        if camera is None:
            camera = CentralCamera.Default(name='')

        P = smbase.circle([0, 0, 3], 1, resolution=3)
        # self.planes = np.tile([0, 0, 1, -3], (3, 1)).T
        pose_0 = SE3(1, 1, -3) * SE3.Rz(0.6)
        pose_d = SE3(0, 0, 1)
        self = cls(camera, P=P, pose_0=pose_0, pose_d=pose_d)
        self.plane = [0, 0, 1, -3]

        return self


    def init(self, pose_d=None):
        """
        Initialize IBVS line simulation.

        Implicitly called by ``run`` to initialize state variables.

        :seealso: :meth:`run` :meth:`VisualServo.init`
        """
        super().init()
        self.camera.clf()

        # final pose is specified in terms of a camera-target pose
        self.f_star_retinal = self.getlines(self.pose_d, np.linalg.inv(self.camera.K)) # in retinal coordinates
        self.f_star = self.getlines(self.pose_d) # in image coordinates


    def step(self, t):
        """
        Compute one timestep of IBVS line simulation.

        :param t: simulation time
        :type t: float
        :return: simulation status, 0 if OK, 1 if terminating
        :rtype: int

        Called by the ``run`` method and performs the following steps:

        * find projections of world lines in current camera view
        * compute the image Jacobian and camera velocity 
        * incrementally update the camera pose.

        :seealso: :meth:`run` :meth:`VisualServo.run`
        """
        status = 0
        Z_est = []
        
        # compute the lines
        f = self.getlines(self.camera.pose)

        # now plot them
        if self.graphics:
            #self.camera.clf()
            colors = 'rgb'
            for i in range(f.shape[1]):
                # plot current line
                self.plot_line_tr(self.camera, f[:, i], color=colors[i])
                # plot demanded line
                self.plot_line_tr(self.camera, self.f_star[:, i], color=colors[i], linestyle='--')

        f_retinal = self.getlines(self.camera.pose, scale=np.linalg.inv(self.camera.K))

        # compute image plane error as a column
        e = f_retinal - self.f_star_retinal   # feature error on retinal plane
        e = e.ravel('F')
        for i in range(0, len(e), 2):
            e[i] = smbase.angdiff(e[i])
    
        J = self.camera.visjac_l(f_retinal, self.plane)

        # compute the velocity of camera in camera frame
        v = -self.lmbda * np.linalg.pinv(J) @ e
        if self.verbose:
            print('v:', v)

        # update the camera pose
        Td = SE3.Delta(v)    # differential motion

        self.camera.pose = self.camera.pose @ Td       # apply it to current pose
        # update the history variables
        hist = self._history()
        hist.f = f.ravel()
        hist.vel = v
        hist.e = e
        hist.enorm = np.linalg.norm(e)
        hist.jcond = np.linalg.cond(J)
        hist.pose = self.camera.pose

        self.history.append(hist)
        if np.linalg.norm(e) < self.eterm:
            status = 1

        return status

    def getlines(self, pose, scale=None):
        """
        Compute image plane lines

        :param pose: the camera viewpoint
        :type pose: SE3 instance
        :return: lines in :math:`(\theta, \rho)` format, one per column
        :rtype: array_like(2,N)

        Consecutive pairs of world points ``P`` passed to the constructor
        define a line in 3D.

        The world points are projected and 2D lines determined in :math:`(\theta, \rho)`
        format.
        """
        # one line per column
        #  row 0 theta
        #  row 1 rho
        # project corner points to image plane
        p = self.camera.project_point(self.P, pose=pose)

        if scale is not None:
            p = smbase.homtrans(scale, p)

        # compute lines and their slope and intercept

        lines = []
        for i in range(p.shape[1]):
            j = (i + 1) % p.shape[1]
            theta = np.arctan2(p[0, j] - p[0, i], p[1, i] - p[1, j])
            rho = np.cos(theta) * p[0, i] + np.sin(theta) * p[1, i]
            lines.append((theta, rho))
        return np.array(lines).T

    @staticmethod
    def plot_line_tr(camera, lines, **kwargs):
    # %CentralCamera.plot_line_tr  Plot line in theta-rho format
    # %
    # % CentralCamera.plot_line_tr(L) plots lines on the camera's image plane that
    # % are described by columns of L with rows theta and rho respectively.
    # %
    # % See also Hough.

        ax = camera._ax
        x = np.r_[ax.get_xlim()]
        y = np.r_[ax.get_ylim()]

        lines = smbase.getmatrix(lines, (2, None))
        # plot it
        for theta, rho in lines.T:
            #print(f'{theta=}, {rho=}')
            if np.abs(np.cos(theta)) > 0.5:
                # horizontalish lines
                ax.plot(x, -x * np.tan(theta) + rho / np.cos(theta), **kwargs)
            else:
                # verticalish lines
                ax.plot(-y / np.tan(theta) + rho / np.sin(theta), y, **kwargs)




class IBVS_e(VisualServo):

    def __init__(self, camera, eterm=0.08, plane=None, lmbda=0.04, **kwargs):
        r"""
        Image-based visual servo for ellipse features class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param P: world points in frame {G} define lines, defaults to None
        :type P: array_like(3,N), optional
        :param pose_d: desired pose of goal {G} with respect to camera, defaults to None
        :type pose_d: SE3 instance, optional
        :param eterm: termination threshold on residual error, defaults to 0.5
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.08
        :type lmbda: float, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param plane: plane parameters :math:`ax+by+cz+d=0`, defaults to None
        :type plane: array_like(4), optional
        :param depthest: run simple depth estimator, defaults to False
        :type depthest: bool, optional
        :param vmax: maximum velocity, defaults to None
        :type vmax: float, optional
        :param smoothstart: enable smooth start with this value as :math:`\mu`, defaults to None
        :type smoothstart: float, optional

        The world lines are defined by consecutive pairs of points in ``P``.

        The goal is defined by the image plane lines as viewed from the pose
        ``pose_d``.  Note that ``pose_d`` is not used by the controller, only
        to obtain the desired image plane lines.

        Example::

                camera = CentralCamera.Default()
                Tc = trnorm( Tc * delta2tr(v) )
                Tc0 = transl(1,1,-3)*trotz(0.6)
                p_f = bsxfun(@plus, 200*[-1 -1 1 1 -1 1 1 -1], cam.pp')
                ibvs = IBVS(cam, 'T0', Tc0, 'p_f', p_f)
                self.run[]
                self.plot_p[]

        The plane applies to all lines.

        References::
            - Robotics, Vision & Control, Chap 15
              P. Corke, Springer 2011.

        .. note::
            - The history property is a vector of structures each of which is a
              snapshot at each simulation step of information about the image
              plane, camera pose, error, Jacobian condition number, error norm,
              image plane size and desired feature locations.
            - This implementation has a sign change compared to the task
              function notation (Chaumette papers), the the error in this code is
              desired-actual which means the control gain is positive.
        """

        
        # invoke superclass constructor
        print('IBVS_e constructor')
        super().__init__(camera, type='point', **kwargs)

        self.eterm = eterm
        self.plane = plane
        self.lmbda = lmbda        

    @classmethod
    def Example(cls, camera=None, **kwargs):
        # run a canned example
        print('canned example, ellipse + point-based IBVS')
        if camera is None:
            camera = CentralCamera.Default(name='')

        self = cls(camera, 
            P=smbase.circle(radius=0.5, centre=[0, 0, 3], resolution=10),
            pose_d=SE3(0.5, 0.5, 1),
            pose_0=SE3(0.5, 0.5, 0) * SE3.Rx(0.3),
            plane = [0, 0, 1, -3],  # plane Z=3
            **kwargs)

        return self

    def init(self):
        """
        Initialize IBVS ellipse simulation.

        Implicitly called by ``run`` to initialize state variables.

        :seealso: :meth:`run` :meth:`VisualServo.init`
        """

        # desired feature coordinates.  This vector comprises the ellipse
        # parameters (5) and the coordinaes of 1 point
        super().init()

        self.f_star = np.r_[
                self.get_ellipse_parameters(self.pose_d),
                self.camera.project_point(self.P[:, 0], pose=self.pose_d).ravel()
            ]
        
        self.ellipse_star = self.camera.project_point(self.P, pose=self.pose_d)
        # self.ellipse_star = self.camera.project([self.P self.P(:,1)], pose=self.pose_d)


    def get_ellipse_parameters(self, pose):
        p = self.camera.project_point(self.P, pose=pose) #, retinal=True)

        # # convert to normalized image-plane coordinates
        p = smbase.homtrans(np.linalg.inv(self.camera.K), p)
        x, y = p

        # solve for the ellipse parameters
        # x^2 + A1 y^2 - 2 A2 xy + 2 A3 x + 2 A4 y + A5 = 0
        A = np.column_stack([y**2, -2*x*y, 2*x, 2*y, np.ones(x.shape)])
        b = -(x**2)
        theta, resid, *_ = np.linalg.lstsq(A, b, rcond=None)         # least squares solution
        return theta

    def step(self, t):
        """
        Compute one timestep of IBVS line simulation.

        :param t: simulation time
        :type t: float
        :return: simulation status, 0 if OK, 1 if terminating
        :rtype: int

        Called by the ``run`` method and performs the following steps:

        * find projections of world lines in current camera view
        * compute the image Jacobian and camera velocity 
        * incrementally update the camera pose.

        :seealso: :meth:`run` :meth:`VisualServo.run`
        """
        
        status = 0
        Z_est = []

        # compute feature vector
        f = np.r_[
                self.get_ellipse_parameters(self.camera.pose),
                self.camera.project_point(self.P[:, 0]).flatten(order="F")
            ]
        
        # compute image plane error as a column
        e = f - self.f_star   # feature error
        
        # compute the Jacobians and stack them
        Je = self.camera.visjac_e(f[:5], self.plane)  # ellipse
        Jp = self.camera.visjac_p(f[5:], -self.plane[3]) # point
        J = np.vstack([Je, Jp])

        # compute the velocity of camera in camera frame
        v = -self.lmbda * np.linalg.pinv(J) @ e

        # update the camera pose
        self.camera.pose @= SE3.Delta(v)

        if self.verbose:
            #print(f"{cond=}, {v=}")
            self.pose.printline()

        # update the history variables
        hist = self._history()
        hist.f = f
        hist.p = self.camera.project_point(self.P)
        hist.vel = v
        hist.e = e
        hist.enorm = np.linalg.norm(e)
        hist.jcond = np.linalg.cond(J)
        hist.pose = self.camera.pose
        self.history.append(hist)
        
        if hist.enorm < self.eterm:
            status = 1

        return status

class IBVS_sph(VisualServo):

    def __init__(self, camera, eterm=0.001, lmbda=0.1, depth=None, **kwargs):
        r"""
        Image-based visual servo for ellipse features class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param P: world points in frame {G} define lines, defaults to None
        :type P: array_like(3,N), optional
        :param pose_d: desired pose of goal {G} with respect to camera, defaults to None
        :type pose_d: SE3 instance, optional
        :param eterm: termination threshold on residual error, defaults to 0.5
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.08
        :type lmbda: float, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param plane: plane parameters :math:`ax+by+cz+d=0`, defaults to None
        :type plane: array_like(4), optional
        :param depthest: run simple depth estimator, defaults to False
        :type depthest: bool, optional
        :param vmax: maximum velocity, defaults to None
        :type vmax: float, optional
        :param smoothstart: enable smooth start with this value as :math:`\mu`, defaults to None
        :type smoothstart: float, optional

        The world lines are defined by consecutive pairs of points in ``P``.

        The goal is defined by the image plane lines as viewed from the pose
        ``pose_d``.  Note that ``pose_d`` is not used by the controller, only
        to obtain the desired image plane lines.

        Example::

                camera = CentralCamera.Default()
                Tc = trnorm( Tc * delta2tr(v) )
                Tc0 = transl(1,1,-3)*trotz(0.6)
                p_f = bsxfun(@plus, 200*[-1 -1 1 1 -1 1 1 -1], cam.pp')
                ibvs = IBVS(cam, 'T0', Tc0, 'p_f', p_f)
                self.run[]
                self.plot_p[]

        The plane applies to all lines.

        References::
            - Robotics, Vision & Control, Chap 15
              P. Corke, Springer 2011.

        .. note::
            - The history property is a vector of structures each of which is a
              snapshot at each simulation step of information about the image
              plane, camera pose, error, Jacobian condition number, error norm,
              image plane size and desired feature locations.
            - This implementation has a sign change compared to the task
              function notation (Chaumette papers), the the error in this code is
              desired-actual which means the control gain is positive.
        """

        # invoke superclass constructor
        super().__init__(camera, type='point', **kwargs)
        
        self.lmbda = lmbda
        self.eterm = eterm
        self.depth = depth
        
                
    @classmethod
    def Example(cls):
        # run a canned example
        print('canned example, spherical IBVS with 4 points');
        if camera is None:
            camera = SphericalCamera.Default(name='')
        self = cls(camera, **kwargs)
        self.P = mkgrid(2, side=1.5, pose=SE3(0, 0, 0.5))
        self.pose_d = SE3(0, 0, -1.5) * SE3.Rz(1)
        self.pose_0 = SE3(0.3, 0.3, -2) * SE3.Rz(0.2)
        # self.T0 = transl(-1,-0.1,-3);%*trotx(0.2)

    def init(self):

        super().init()

        # final pose is specified in terms of a camera-target pose
        #   convert to image coords
        self.p_star = self.camera.project_point(self.P, pose=self.pose_d)


    def step(self, t):
        """
        Compute one timestep of IBVS line simulation.

        :param t: simulation time
        :type t: float
        :return: simulation status, 0 if OK, 1 if terminating
        :rtype: int

        Called by the ``run`` method and performs the following steps:

        * find projections of world lines in current camera view
        * compute the image Jacobian and camera velocity 
        * incrementally update the camera pose.

        :seealso: :meth:`run` :meth:`VisualServo.run`
        """
        status = 0;
        Z_est = [];
        
        # compute image plane error as a column
        p = self.camera.project_point(self.P)  # (phi, theta)
        # if self.verbose:
        #     print(f"{p=}")

        e = self.p_star - p   # feature error
        e[0, :] = smbase.wrap_mpi_pi(e[0, :])
        e[1, :] = smbase.wrap_0_pi(e[1, :])
        e = e.flatten(order='F')
    
        # compute the Jacobian
        if self.depth is None:
            # exact depth from simulation (not possible in practice)
            P_C = self.camera.pose.inv() * self.P
            J = self.camera.visjac_p(p, P_C[2, :])
        else:
            J = self.camera.visjac_p(pt, self.depth)

        # compute the velocity of camera in camera frame
        try:
            v = self.lmbda * np.linalg.pinv(J) @ e
        except np.linalg.LinAlgError:
            status = -1

        # if self.verbose:
        #     print(f"{v=}")

        # update the camera pose
        self.camera.pose @= SE3.Delta(v) 

        # draw lines from points to centre of camera
        if self.graphics:
            centre = self.camera.pose.t
            plt.sca(self.ax_3dview)
            for P in self.P.T:
                plt.plot(*[(centre[i], P[i]) for i in range(3)], 'k')

        # update the history variables
        hist = self._history()
        hist.p = p
        hist.vel = v
        hist.e = e
        hist.enorm = np.linalg.norm(e)
        hist.jcond = np.linalg.cond(J)
        hist.pose = self.camera.pose
        self.history.append(hist)

        if hist.enorm < self.eterm:
            status = 1
        return status

    def plot_p(self):
        # result is a vector with row per time step, each row is u1, v1, u2, v2 ...
        for i in range(self.npoints):
            u = [h.p[0, i] for h in self.history]  # get data for i'th point
            v = [h.p[1, i] for h in self.history]
            plt.plot(u, v, 'b')
        
        # mark the initial target shape
        smbase.plot_point(self.history[0].p, 'o', markeredgecolor='k', markerfacecolor='w', label='initial')
        
        # mark the goal target shape
        smbase.plot_point(self.p_star, 'k*', markeredgecolor='k', markerfacecolor='k', label='goal')

        # axis([0 self.camera.npix[0] 0 self.camera.npix[1]])
        # daspect([1 1 1])
        ax = plt.gca()

        plt.grid(True)
        ax.set_xlabel('Azimuth φ (rad)')
        ax.set_ylabel('Colatitude θ (rad)')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, np.pi)
        ax.invert_yaxis()
        plt.legend(loc='lower right')
        ax.set_facecolor('lightyellow')


# %IBVS   Implement classical IBVS for point features
# %
# %  results = ibvs(T)
# %  results = ibvs(T, params)
# %
# %  Simulate IBVS with for a square target comprising 4 points is placed 
# %  in the world XY plane. The camera/robot is initially at pose T and is
# %  driven to the orgin.
# %
# %  Two windows are shown and animated:
# %   1. The camera view, showing the desired view (*) and the 
# %      current view (o)
# %   2. The external view, showing the target points and the camera
# %
# % The results structure contains time-history information about the image
# % plane, camera pose, error, Jacobian condition number, error norm, image
# % plane size and desired feature locations.
# %
# % The params structure can be used to override simulation defaults by
# % providing elements, defaults in parentheses:
# %
# %   target_size    - the side length of the target in world units (0.5)
# %   target_center  - center of the target in world coords (0,0,3)
# %   niter          - the number of iterations to run the simulation (500)
# %   eterm          - a stopping criteria on feature error norm (0)
# %   lambda         - gain, can be scalar or diagonal 6x6 matrix (0.01)
# %   ci             - camera intrinsic structure (camparam)
# %   depth          - depth of points to use for Jacobian, scalar for
# %                    all points, of 4-vector.  If null take actual value
# %                    from simulation      ([])
# %
# % SEE ALSO: ibvsplot

# % IMPLEMENTATION NOTE
# %
# % 1.  As per task function notation (Chaumette papers) the error is
# %     defined as actual-demand, the reverse of normal control system
# %     notation.
# % 2.  The gain, lambda, is always positive
# % 3.  The negative sign is written into the control law

class IBVS_polar(VisualServo):

    def __init__(self, camera, eterm=0.01, lmbda=0.02, depth=None, **kwargs):
        r"""
        Image-based visual servo for ellipse features class

        :param camera: central camera mode
        :type camera: CentralCamera instance
        :param P: world points in frame {G} define lines, defaults to None
        :type P: array_like(3,N), optional
        :param pose_d: desired pose of goal {G} with respect to camera, defaults to None
        :type pose_d: SE3 instance, optional
        :param eterm: termination threshold on residual error, defaults to 0.5
        :type eterm: float, optional
        :param lmbda: positive control gain, defaults to 0.08
        :type lmbda: float, optional
        :param pose_0: initial camera pose, overrides pose of camera object, defaults to None
        :type pose_0: SE3 instance, optional
        :param plane: plane parameters :math:`ax+by+cz+d=0`, defaults to None
        :type plane: array_like(4), optional
        :param depthest: run simple depth estimator, defaults to False
        :type depthest: bool, optional
        :param vmax: maximum velocity, defaults to None
        :type vmax: float, optional
        :param smoothstart: enable smooth start with this value as :math:`\mu`, defaults to None
        :type smoothstart: float, optional

        The world lines are defined by consecutive pairs of points in ``P``.

        The goal is defined by the image plane lines as viewed from the pose
        ``pose_d``.  Note that ``pose_d`` is not used by the controller, only
        to obtain the desired image plane lines.

        Example::

                camera = CentralCamera.Default()
                Tc = trnorm( Tc * delta2tr(v) )
                Tc0 = transl(1,1,-3)*trotz(0.6)
                p_f = bsxfun(@plus, 200*[-1 -1 1 1 -1 1 1 -1], cam.pp')
                ibvs = IBVS(cam, 'T0', Tc0, 'p_f', p_f)
                self.run[]
                self.plot_p[]

        The plane applies to all lines.

        References::
            - Robotics, Vision & Control, Chap 15
              P. Corke, Springer 2011.

        .. note::
            - The history property is a vector of structures each of which is a
              snapshot at each simulation step of information about the image
              plane, camera pose, error, Jacobian condition number, error norm,
              image plane size and desired feature locations.
            - This implementation has a sign change compared to the task
              function notation (Chaumette papers), the the error in this code is
              desired-actual which means the control gain is positive.
        """
        # monkey patch the plot setup for the CentralCamera object
        import types
        camera._init_imageplane = types.MethodType(self._init_imageplane, camera)
        camera._project_point = camera.project_point
        camera.project_point = types.MethodType(self._project_polar, camera)

        # invoke superclass constructor
        super().__init__(camera, type='point', **kwargs)
        
        self.lmbda = lmbda
        self.eterm = eterm
        self.depth = depth


    def init(self):

        # initialize the vservo variables
        super().init()

        # if 0 % isempty(self.h_rt) || ~ishandle(self.h_rt)
        #     fprintf('create rt axes\n');
        #     self.h_rt = axes;
        #     set(self.h_rt, 'XLimMode', 'manual');
        #     set(self.h_rt, 'YLimMode', 'manual');
        #     set(self.h_rt, 'NextPlot', 'replacechildren');
        #     axis([-pi pi 0 sqrt(2)])
        #     %axis([-pi pi 0 norm(self.camera.npix-self.camera.pp)])
        #     xlabel('\theta (rad)');
        #     ylabel('r (pix)');
        #     title('polar coordinate feature space');
        #     grid
        # end
        # %axes(self.h_rt)
        # %cla


        # final pose is specified in terms of a camera-target pose
        #  convert to image coords
        self.th_r_star = self.camera.project_point(self.P, pose=self.pose_d)

        self.plot_point(self.p_star, '*')

        if smbase.isscalar(self.lmbda):
            self.lmbda = np.diag([self.lmbda] * 6)

        # show the reference location, this is the view we wish to achieve
        # when Tc = Tct_star
        # if 0
        # self.camera.clf()
        # self.camera.plot(self.p_star, '*'); % create the camera view
        # self.camera.hold(true);
        # self.camera.plot(self.P, 'pose', self.T0, 'o'); % create the camera view
        # pause(2)
        # self.camera.hold(false);
        # self.camera.clf();
        # end

        # %self.camera.plot(self.P);    % show initial view

        # % this is the 'external' view of the points and the camera
        # %plot_sphere(self.P, 0.05, 'b')
        # %cam2 = showcamera(T0);
        # clf
        # self.camera.plot_camera(self.P, 'label');
        # # %camup([0,-1,0]);

        self.history = [];


    def step(self, t):
        status = 0;
        Zest = [];
        
        hist = self._history()

        # compute the polar projection view (phi, r)
        p = self.camera.project_point(self.P)

        # compute image plane error as a column
        e = self.p_star - p  # feature error

        e[0, :] = smbase.wrap_mpi_pi(e[0, :])
        e = e.flatten(order='F')  # convert columnwise to a 1D vector 
        
        # compute the Jacobian
        if self.depth is None:
            # exact depth from simulation (not possible in practice)
            pt = self.camera.pose.inv() * self.P
            J = self.camera.visjac_p_polar(p, pt[2, :])
        else:
            # use the default depth
            J = self.camera.visjac_p_polar(p, self.depth)

        # compute the velocity of camera in camera frame
        try:
            v = -self.lmbda @ np.linalg.pinv(J) @ e
        except np.linalg.LinAlgError:
            return -1

        if self.verbose:
            print(v)

        vmax = 0.02
        if np.linalg.norm(v) > vmax:
            v = smbase.unitvec(v) * vmax

        # update the camera pose
        Td = SE3.Delta(v) # differential motion
        # Td = SE3(trnorm(delta2tr(v)))    
        #Td = expm( skewa(v) )
        #Td = SE3( delta2tr(v) )
        self.camera.pose @= Td       # apply it to current pose

        # update the history variables
        hist.p = p
        vel = Td.delta()
        hist.vel = vel
        hist.e = e
        hist.enorm = np.linalg.norm(e)
        hist.jcond = np.linalg.cond(J)
        hist.pose = self.camera.pose

        self.history.append(hist)

        if np.linalg.norm(e) < self.eterm:
            status = 1

        return status

    def plot_p(self):
        # result is a vector with row per time step, each row is u1, v1, u2, v2 ...
        for i in range(self.npoints):
            u = [h.p[0, i] for h in self.history]  # get data for i'th point
            v = [h.p[1, i] for h in self.history]
            plt.plot(u, v, 'b')
        
        # mark the initial target shape
        smbase.plot_point(self.history[0].p, 'o', markeredgecolor='k', markerfacecolor='w', label='initial')
        
        # mark the goal target shape
        smbase.plot_point(self.p_star, 'k*', markeredgecolor='k', markerfacecolor='k', label='goal')

        # axis([0 self.camera.npix[0] 0 self.camera.npix[1]])
        # daspect([1 1 1])
        ax = plt.gca()

        plt.grid(True)
        ax.set_xlabel('Azimuth φ (rad)')
        ax.set_ylabel('normalized radius r')
        ax.set_xlim(-np.pi, np.pi)
        rmax = np.linalg.norm(np.r_[self.camera.width, self.camera.height] - self.camera.pp) * 2 / self.camera.width
        ax.set_ylim(0, rmax)
        plt.legend()
        ax.set_facecolor('lightyellow')

    @staticmethod
    def _project_polar(self, P, pose=None, objpose=None):
        # bound to project_point()

        # overloaded projection method, projects to polar coordinates

        p = self._project_point(P, pose=pose, objpose=objpose)
        # %p = homtrans( inv(self.camera.K), p);

        pp = self.pp
        u = p[0, :] - pp[0]
        v = p[1, :] - pp[1]
        th_r = np.array([np.arctan2(v,u),
                         np.sqrt(u**2 + v**2) / self.width * 2])

        # %line(rt(:,2), rt(:,1), 'Marker', 'o', 'MarkerFaceColor', 'k', 'Parent', self.h_rt)
        # % plot points on rt plane

        return th_r

    @staticmethod
    def _init_imageplane(self, fig=None, ax=None):
        if self._new_imageplane(fig, ax):
            return

        ax = self._ax

        ax.set_xlim(-np.pi, np.pi)
        rmax = np.linalg.norm(np.r_[self.width, self.height] - self.pp) * 2 / self.width
        ax.set_ylim(0, rmax)

        ax.autoscale(False)
        ax.grid(True)

        ax.set_xlabel('Azimuth φ (rad)')
        ax.set_ylabel('normalized radius')

        ax.set_title(self.name)
        ax.set_facecolor('lightyellow')
        ax.figure.canvas.set_window_title('Machine Vision Toolbox for Python')

        # TODO figure out axes ticks, etc
        return ax  # likely this return is not necessary
