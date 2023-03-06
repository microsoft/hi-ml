#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# Make the source code for both packages available here
sys.path.insert(0, os.path.abspath('../../hi-ml/src'))
sys.path.insert(0, os.path.abspath('../../hi-ml-azure/src'))
sys.path.insert(0, os.path.abspath('../../hi-ml-cpath/src'))
sys.path.insert(0, os.path.abspath('../../hi-ml-multimodal/src'))

# -- Project information -----------------------------------------------------

project = 'hi-ml'
copyright = '2021, InnerEye'
author = 'InnerEye'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_automodapi.automodapi',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    "sphinxarg.ext",
]

numpydoc_show_class_members = False

# napoleon_google_docstring = True
# napoleon_use_param = False
# napoleon_use_rtype = True
# napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # type: ignore

modindex_common_prefix = ["health_azure.", "health_ml.", "health_cpath."]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# Add packages here that are expensive to import during doc generation, like pytorch
autodoc_mock_imports = [""]

# This is the default language for syntax highlighting for all files that in included in .rst files
highlight_language = "python"

# For classes, insert documentation from the class itself AND the constructor
autoclass_content = "both"

autodoc_default_options = {
    'members': True,
}
