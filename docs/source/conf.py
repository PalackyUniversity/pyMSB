# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "pyMSB"
copyright = "2024, J. Dostál, V. Skoumal"
author = "J. Dosál, V. Skoumal"
version = ""
release = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

pygments_style = "sphinx"

templates_path = ["_templates"]
exclude_patterns = []

numpydoc_show_class_members = False
autosummary_generate = True
autodoc_typehints = "none"
remove_from_toctrees = ["**/classmethods/*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = "pyMSB"

html_theme_options = {
    "show_nav_level": 1,
    "navbar_start": ["navbar-logo"],
}
