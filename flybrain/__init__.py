"""Fly brain simulator — modular package.

Split from the original monoliths (``app.py`` ~311 lines, ``server.py``
~1124 lines, ``flygym_bridge.py`` ~533 lines) into single-responsibility
modules. Thin shims (``app.py`` / ``server.py`` / ``flygym_bridge.py``)
re-export from here so existing run commands keep working.
"""

__version__ = "2.0.0"
