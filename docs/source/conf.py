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

sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/model/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/model/architecture/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/model/one_hot_conversion/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/preprocessing/homography_converter/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/preprocessing/ipm/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/Cam2BEV/preprocessing/occlusion/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/SemSeg/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/SIN/src/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/SIN/src/data/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/SIN/src/models/arch/'))
sys.path.insert(0, os.path.abspath('../../DySi_Select_Simulation/SIN/src/utils/'))


# -- Project information -----------------------------------------------------

project = 'DySi_Select'
copyright = '2023, Akshay Narla, IAS'
author = 'Akshay Narla'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_simplepdf"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

html_theme_options = {
    "sidebar_hide_name": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


simplepdf_vars = {
    'primary': '#6495ED',
    'secondary': '#00BFFF',
    'cover': '#ffffff',
    'white': '#ffffff',
    'links': '#6495ED',
    'cover-overlay': 'rgba(250, 35, 35, 0.5)',
    'top-left-content': 'counter(page)',
    'bottom-center-content': '"2023, Akshay Narla, IAS"',
}