from machinevisiontoolbox import CentralCamera
camera = CentralCamera.Default()
camera.plot_point([0.2, 0.3, 2])
camera.plot_point([0.2, 0.3, 2], 'r*')
camera.plot_point([0.2, 0.3, 2], pose=SE3(0.1, 0, 0))