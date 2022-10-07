from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

req = [
    "numpy",
    "scipy",
    "matplotlib",
    "opencv-python",
    "spatialmath-python",
    "pgraph-python",
    "ansitable",
    "mvtb-data"
    ]

docs_req = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-autorun",
]

dev_req = ["pytest", "pytest-cov", "flake8", "pyyaml"]

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the release/version string
with open(os.path.join(here, "RELEASE"), encoding="utf-8") as f:
    release = f.read()

setup(
    name="machinevision-toolbox-python",
    version=release,
    description="A machine vision for education and research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petercorke/machinevision-toolbox-python",
    author="Peter Corke and Dorian Tsai",
    license="MIT",
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    project_urls={
        "Documentation": "https://petercorke.github.io/machinevision-toolbox-python",
        "Source": "https://github.com/petercorke/machinevision-toolbox-python",
        "Tracker": "https://github.com/petercorke/machinevision-toolbox-python/issues",
        "Coverage": "https://codecov.io/gh/petercorke/machinevision-toolbox-python",
    },
    keywords="python machine-vision computer-vision multiview-geometry features color blobs",
    packages=find_packages(),
    install_requires=req,
    extras_require={
        "docs": docs_req,
        "dev": dev_req,
    },
    data_files=[('.', ['RELEASE'])],
)


