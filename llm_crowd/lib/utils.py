import os
from pathlib import Path

ROOTDIR = Path(os.path.realpath(__file__)).parent.parent.parent
DATADIR = ROOTDIR / 'data'
CONFDIR = ROOTDIR / 'config'
