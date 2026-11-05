"""Tests for M3DC1Dataset class."""

from pathlib import Path

import pytest

from surge.datasets import M3DC1Dataset

# The M3D-C1 HDF5 readers are facility-specific and not part of the public
# tree; these tests only run where they are available (SURGE_M3DC1_SCRIPTS or
# an ignored scripts/m3dc1/ checkout).
def _m3dc1_readers_available() -> bool:
    try:
        from surge.datasets.m3dc1 import _get_m3dc1_loaders
        _get_m3dc1_loaders()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _m3dc1_readers_available(),
                                reason="facility-specific M3D-C1 readers not available")



def test_m3dc1_dataset_from_batch_dir_empty_raises(tmp_path):
    """from_batch_dir raises FileNotFoundError when no complex_v2 files exist."""
    with pytest.raises(FileNotFoundError, match="No .* found"):
        M3DC1Dataset.from_batch_dir(tmp_path, verbose=False)


def test_m3dc1_dataset_from_batch_dir_nonexistent():
    """from_batch_dir raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError):
        M3DC1Dataset.from_batch_dir("/nonexistent/batch_999", verbose=False)


def test_m3dc1_dataset_from_sdata_complex_v2_nonexistent():
    """from_sdata_complex_v2 raises FileNotFoundError for nonexistent path."""
    with pytest.raises(FileNotFoundError, match="sdata_complex_v2"):
        M3DC1Dataset.from_sdata_complex_v2("/nonexistent/sdata_complex_v2.h5")


def test_m3dc1_dataset_class_inherits_surrogate_dataset():
    """M3DC1Dataset is a subclass of SurrogateDataset."""
    from surge.dataset import SurrogateDataset
    assert issubclass(M3DC1Dataset, SurrogateDataset)
