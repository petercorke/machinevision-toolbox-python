def version():
    """
    OpenCV and Machine Vision Toolbox version.

    Displays the versions of the Machine Vision Toolbox for Python, Spatial Math Toolbox for Python, NumPy, and OpenCV.
    Also displays the OpenCV build information.

    :seealso: :func:`cv2.getBuildInformation`
    """
    import cv2
    import machinevisiontoolbox as mvt
    import spatialmath as sm
    import numpy as np
    import textwrap

    print("Machine Vision Toolbox for Python: version", mvt.__version__)
    print("Spatial Math Toolbox for Python: version  ", sm.__version__)
    print("NumPy version:                            ", np.__version__)
    print("OpenCV version:                           ", cv2.__version__)

    print("\nOpenCV build information:")
    print(textwrap.indent(cv2.getBuildInformation(), "> ", lambda line: True))


if __name__ == "__main__":
    version()
