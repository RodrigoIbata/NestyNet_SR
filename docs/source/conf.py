# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'NestyNet_SR'
copyright = '2023-2026, Rodrigo Ibata'
author = 'Rodrigo Ibata'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}


# ---------------------------------------------------------------------------
# Autodoc docstring sanitization
# ---------------------------------------------------------------------------
# The research docstrings use non-ASCII glyphs, premature Sphinx roles, and
# norm-style bars that docutils tries (and fails) to parse.  Rather than rewrite
# every docstring, render them as plain literal text blocks so the build stays
# clean.  (Mirrors the NestyNet docs configuration.)

_PY_ROLE_RE = re.compile(r":py[a-z_]*:`([^`]+)`")


def _sanitize_autodoc_lines(lines):
    out = []
    for ln in lines:
        s = ln.expandtabs(4)
        # Common non-ASCII bullets/arrows used in code comments/docstrings.
        s = s.replace("•", "*").replace("→", "->").replace("−", "-")
        # Some docstrings use Sphinx roles that docutils may see too early.
        s = _PY_ROLE_RE.sub(r"``\1``", s)
        # Escape norm-like bars so docutils does not interpret them as substitutions.
        s = re.sub(r"(?<!\\)\|([^|]+)\|", r"\\|\1\\|", s)
        out.append(s)
    return out


def _literalize_docstring(lines):
    """Render docstrings as plain text blocks to avoid parser failures."""
    if not lines:
        return []
    body = [ln.rstrip() for ln in lines]
    if not any(ln.strip() for ln in body):
        return []
    lit = [".. code-block:: text", ""]
    for ln in body:
        lit.append(("   " + ln) if ln else "   ")
    return lit


def _autodoc_process_docstring(_app, _what, _name, _obj, _options, lines):
    sanitized = _sanitize_autodoc_lines(list(lines))
    lines[:] = _literalize_docstring(sanitized)


def setup(app):
    app.connect("autodoc-process-docstring", _autodoc_process_docstring)
