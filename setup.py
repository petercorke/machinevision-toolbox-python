from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the release/version string
with open(os.path.join(here, 'RELEASE'), encoding='utf-8') as f:
    release = f.read()

# list all data folders here, to ensure they get packaged

data_folders = [
    'machinevisiontoolbox/data',
    'machinevisiontoolbox/images',
]


def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        # skip bulky image folders, PyPI has 100MB limit :()
        if any([folder in pathhere for folder in ['bridge', 'campus', 'mosaic']]):
            continue
        for filename in filenames:
            paths.append(os.path.join('..', pathhere, filename))
    return paths

extra_files = []
for data_folder in data_folders:
    extra_files += package_files(data_folder)

extra_files.append('../RELEASE')

req = [
    'numpy',
    'scipy',
    'matplotlib',
    'opencv-python',
    'spatialmath-python',
    'ansitable',
    ]

docs_req = [
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx-autorun',
]

setup(
    name='machinevision-toolbox-python', 

    version=release,

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    description='Machine vision capability for Python.', #TODO
    
    long_description=long_description,
    long_description_content_type='text/markdown',

    classifiers=[
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    # Pick your license as you wish (should match "license" above)
     'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3 :: Only'],

    project_urls={
        'Documentation': 'https://petercorke.github.io/machinevision-toolbox-python',
        'Source': 'https://github.com/petercorke/machinevision-toolbox-python',
        'Tracker': 'https://github.com/petercorke/machinevision-toolbox-python/issues',
        'Coverage': 'https://codecov.io/gh/petercorke/machinevision-toolbox-python'
    },

    url='https://github.com/petercorke/machinevision-toolbox-python',

    author='Dorian Tsai and Peter Corke',

    author_email='rvc@petercorke.com', #TODO

    keywords='python machine-vision computer-vision color blobs',

    license='MIT',

    python_requires='>=3.6',

    package_data={'machinevisiontoolbox': extra_files},

    packages=find_packages(exclude=["test_*", "TODO*"]),

    install_requires=req,

    extras_require={
        'docs': docs_req,
    }
    
)

# from setuptools.command.install import install
# class InstallWrapper(install):

#   def run(self):
#     # Run the standard PyPi copy
#     install.run(self)
#     # post install stuff here
#     test if images are around
#     install them from some server
#     need an image path
#     how to handle install option [images]??

# # in setup
# cmdclass={'install': InstallWrapper},

