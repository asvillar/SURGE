"""Importing surge.viz must not change the matplotlib backend."""

from __future__ import annotations

import os
import subprocess
import sys


def test_import_surge_viz_keeps_backend():
    code = (
        "import matplotlib; matplotlib.use('svg'); "
        "import surge.viz, surge.viz.animate, surge.viz.campaign; "
        "print(matplotlib.get_backend())"
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1] == "svg"
