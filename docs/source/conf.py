# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('../../machinevisiontoolbox'))
sys.path.append(os.path.abspath('exts'))
print(os.path.abspath('exts'))


# -- Project information -----------------------------------------------------

project = 'Machine Vision Toolbox'
copyright = '2020-, Peter Corke'
author = 'Peter Corke and Dorian Tsai'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.inheritance_diagram',
    'sphinx_autorun',
    "sphinx.ext.intersphinx",
    'blockname',
]

#autoclass_content = 'both' # use __init__ or class docstring
add_function_parentheses = False

# options for spinx_autorun, used for inline examples
#  choose UTF-8 encoding to allow for Unicode characters, eg. ansitable
#  Python session setup, turn off color printing for SE3, set NumPy precision
autorun_languages = {}
autorun_languages['pycon_output_encoding'] = 'UTF-8'
autorun_languages['pycon_input_encoding'] = 'UTF-8'
autorun_languages['pycon_runfirst'] = """
from spatialmath import SE3
SE3._color = False
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from ansitable import ANSITable
ANSITable._color = False
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
autosummary_generate = True
autodoc_member_order = 'bysource'
autosummary_imported_members = True
add_module_name = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['test_*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'github_user': 'petercorke',
    #'github_repo': 'spatialmath-python',
    #'logo_name': False,
    'logo_only': False,
    #'description': 'Spatial maths and geometry for Python',
    'display_version': True,
    'prev_next_buttons_location': 'both',

    }
html_logo = '../../figs/VisionToolboxLogo_CircBlack.png'
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_last_updated_fmt = '%d-%b-%Y'

# see https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx
mathjax3_config = {
    "tex": {
        "macros": {
            # RVC Math notation
            #  - not possible to do the if/then/else approach
            #  - subset only
            #"presup": [r"\,{}^{\scriptscriptstyle #1}\!", 1],
            "presup": [r"\,{}^{#1}\!", 1],
            # groups
            "SE": [r"\mathbf{SE}(#1)", 1],
            "SO": [r"\mathbf{SO}(#1)", 1],
            "se": [r"\mathbf{se}(#1)", 1],
            "so": [r"\mathbf{so}(#1)", 1],
            # vectors
            "vec": [r"\boldsymbol{#1}", 1],
            "dvec": [r"\dot{\boldsymbol{#1}}", 1],
            "hvec": [r"\tilde{\boldsymbol{#1}}", 1],
            "ddvec": [r"\ddot{\boldsymbol{#1}}", 1],
            "fvec": [r"\presup{#1}\boldsymbol{#2}", 2],
            "fdvec": [r"\presup{#1}\dot{\boldsymbol{#2}}", 2],
            "fddvec": [r"\presup{#1}\ddot{\boldsymbol{#2}}", 2],
            "norm": [r"\Vert #1 \Vert", 1],
            # matrices
            "mat": [r"\mathbf{#1}", 1],
            "fmat": [r"\presup{#1}\mathbf{#2}", 2],
            # skew matrices
            "sk": [r"\left[#1\right]", 1],
            "skx": [r"\left[#1\right]_{\times}", 1],
            "vex": [r"\vee\left( #1\right)", 1],
            "vexx": [r"\vee_{\times}\left( #1\right)", 1],
            # quaternions
            "q": r"\mathring{q}",
            "fq": [r"\presup{#1}\mathring{q}", 1],
        }
   }
}

intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "open3d": ("http://www.open3d.org/docs/release/", None),
    'opencv' : ('http://docs.opencv.org/2.4/', None),
    "smtb": ("https://petercorke.github.io/spatialmath-python/", None),
    "pgraph": ("https://petercorke.github.io/pgraph-python/", None),
}
# maybe issues with cv2 https://stackoverflow.com/questions/30939867/how-to-properly-write-cross-references-to-external-documentation-with-intersphin
