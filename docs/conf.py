# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect
import importlib

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "msgjson"
copyright = ""
author = "Zachary Morgan"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zjmorgan/magnetism-tools",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "show_nav_level": 2,
}

# -- Extension configuration -------------------------------------------------

add_module_names = False

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

numpydoc_show_class_members = False


def linkcode_resolve(domain, info):
    baseurl = "https://github.com/zjmorgan/magnetism-tools/blob/main/src/{}.py"
    if "py" not in domain:
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    url = baseurl.format(filename)
    try:
        mod = importlib.import_module(info["module"])
    except ImportError:
        return url
    if hasattr(mod, "__pyx_unpickle_Enum"):
        url += "x"
    objname, *attrname = info["fullname"].split(".")
    obj = getattr(mod, objname, None)
    if obj is None:
        return url
    if attrname:
        for attr in attrname:
            obj = getattr(obj, attr, None)
            if obj is None:
                return url
    try:
        lines = inspect.getsourcelines(obj)
        start, stop = lines[1], lines[1] + len(lines[0]) - 1
        return "{}#L{}-L{}".format(url, start, stop)
    except (TypeError, OSError):
        return url
