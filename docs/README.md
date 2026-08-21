# NestyNet_SR Documentation

This directory contains the Sphinx documentation for NestyNet_SR.

## Building the Documentation

### Install Dependencies

First, install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install Sphinx directly:

```bash
pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints
```

### Build HTML Documentation

#### On Unix/macOS:

```bash
cd docs
make html
```

#### On Windows:

```bash
cd docs
make.bat html
```

### View the Documentation

The built documentation will be in `docs/build/html/`. Open `index.html` in your browser:

```bash
# macOS
open build/html/index.html

# Linux
xdg-open build/html/index.html
```

### Clean Build Files

```bash
make clean
```

## Live Reload (Optional)

For live reloading during development:

```bash
pip install sphinx-autobuild
make livehtml
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── getting_started.rst  # Getting started guide
│   ├── tutorial.rst         # End-to-end tutorial
│   ├── development.rst      # Development guide
│   ├── api/                 # API reference
│   │   ├── index.rst
│   │   ├── sr_core.rst
│   │   ├── sr_search.rst
│   │   ├── sr_de.rst
│   │   ├── sr_gs.rst
│   │   ├── sr_expr_ir.rst
│   │   ├── stat_selection.rst
│   │   ├── discovery.rst
│   │   └── adaptors.rst
│   ├── user_guide/          # User guides
│   │   ├── index.rst
│   │   ├── symbolic_regression.rst
│   │   ├── factorized_search_closure_machine.rst
│   │   ├── de_discovery.rst
│   │   ├── vector_de.rst
│   │   ├── hamiltonian_discovery.rst
│   │   ├── poisson_geometry.rst
│   │   ├── dimensional_analysis.rst
│   │   ├── generalized_symmetries.rst
│   │   └── nonlinear_de_symmetries.rst
│   └── examples/            # Examples
│       └── index.rst
├── statistical_selection.md # Statistical model selection design note (Markdown)
├── Makefile                 # Build script (Unix/macOS)
├── make.bat                 # Build script (Windows)
└── README.md                # This file
```

## Additional Documentation

`docs/statistical_selection.md` documents the archive-conditional statistical
model selection layer (`nestynet_sr.stat_selection`): frozen candidate
archives, common-domain loss audits, and simultaneous confidence Pareto
fronts. It is a standalone Markdown document and is not part of the Sphinx
toctree; read it directly. The corresponding API reference is in
`docs/source/api/stat_selection.rst`.
