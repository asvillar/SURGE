"""M3DC1 dataset loader for SURGE."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

from ..dataset import SurrogateDataset


def _get_m3dc1_loaders() -> tuple[Any, Any]:
    """Import M3DC1 loader modules. Raises ImportError if unavailable."""
    # The M3D-C1 HDF5 readers are facility-specific and are not distributed
    # with SURGE. Point SURGE_M3DC1_SCRIPTS at a directory containing
    # dataset_complex_v2.py / eigenmode_features.py (a checkout may also keep
    # them under scripts/m3dc1/, which is ignored by git).
    env = os.environ.get("SURGE_M3DC1_SCRIPTS")
    scripts_m3dc1 = (Path(env) if env
                     else Path(__file__).resolve().parent.parent.parent / "scripts" / "m3dc1")
    if not scripts_m3dc1.exists():
        raise ImportError(
            "M3D-C1 loader scripts not found. They are facility-specific and not "
            "part of the surge-ml distribution: set SURGE_M3DC1_SCRIPTS to the "
            "directory holding dataset_complex_v2.py and eigenmode_features.py."
        )
    if str(scripts_m3dc1) not in sys.path:
        sys.path.insert(0, str(scripts_m3dc1))

    try:
        from dataset_complex_v2 import load_complex_v2_for_surge
    except ImportError as e:
        raise ImportError(
            "Could not import load_complex_v2_for_surge (dataset_complex_v2.py in SURGE_M3DC1_SCRIPTS). "
            "Ensure h5py is installed: pip install h5py"
        ) from e

    try:
        from loader import convert_sdata_complex_v2_to_dataframe
    except ImportError as e:
        raise ImportError(
            "Could not import convert_sdata_complex_v2_to_dataframe (loader.py in SURGE_M3DC1_SCRIPTS). "
            "Ensure h5py is installed: pip install h5py"
        ) from e

    return load_complex_v2_for_surge, convert_sdata_complex_v2_to_dataframe


class M3DC1Dataset(SurrogateDataset):
    """
    M3DC1-specific dataset loader for delta p spectra and perturbed fields.

    Subclasses SurrogateDataset and provides constructors for:
    - from_batch_dir: per-run sdata_pertfields_grid_complex_v2.h5 under batch_N/run*/sparc_*/
    - from_sdata_complex_v2: single aggregated sdata_complex_v2.h5 file
    """

    @classmethod
    def from_batch_dir(
        cls,
        batch_dir: Union[str, Path],
        *,
        run_pattern: str = "run*",
        sparc_pattern: str = "sparc_*",
        spectrum_field: str = "p",
        spectrum_time_idx: int = -1,
        use_magnitude: bool = True,
        max_cases: Optional[int] = None,
        include_eigenmodes: bool = False,
        target_shape: Optional[tuple] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "M3DC1Dataset":
        """
        Load M3DC1 dataset from a batch directory with per-run complex_v2 HDF5 files.

        Parameters
        ----------
        batch_dir : str or Path
            Root batch directory (e.g. /path/to/batch_16)
        run_pattern : str
            Glob for run directories (default: run*)
        sparc_pattern : str
            Glob for sparc directories (default: sparc_*)
        spectrum_field : str
            Spectrum group name (default: "p" for delta p)
        spectrum_time_idx : int
            Time slice index for spectrum (default: -1 = last)
        use_magnitude : bool
            If True, use |spec| for complex spectrum
        max_cases : int, optional
            Limit number of cases (for testing)
        include_eigenmodes : bool
            If True, add eigenmode_amp_* from C1.h5 when m3dc1/fpy available
        target_shape : (n_modes, n_psi), optional
            If set, resample all cases to this fixed shape (same m-spectrum resolution)
        verbose : bool
            Print progress
        **kwargs
            Passed to build_dataframe_from_batch (e.g. filename="sdata_complex_v2.h5")

        Returns
        -------
        M3DC1Dataset
            Loaded dataset with df, input_columns, output_columns set
        """
        load_complex_v2_for_surge, _ = _get_m3dc1_loaders()
        df, input_cols, output_cols = load_complex_v2_for_surge(
            batch_dir,
            run_pattern=run_pattern,
            sparc_pattern=sparc_pattern,
            spectrum_field=spectrum_field,
            spectrum_time_idx=spectrum_time_idx,
            use_magnitude=use_magnitude,
            max_cases=max_cases,
            include_eigenmodes=include_eigenmodes,
            target_shape=target_shape,
            verbose=verbose,
            **kwargs,
        )
        dataset = cls()
        dataset.load_from_dataframe(
            df,
            input_columns=input_cols,
            output_columns=output_cols,
        )
        dataset.file_path = Path(batch_dir)
        return dataset

    @classmethod
    def from_batch_dir_per_mode(
        cls,
        batch_dir: Union[str, Path],
        *,
        run_pattern: str = "run*",
        sparc_pattern: str = "sparc_*",
        spectrum_field: str = "p",
        spectrum_time_idx: int = -1,
        use_magnitude: bool = True,
        max_cases: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "M3DC1Dataset":
        """
        Load in per-mode format: one row per (case, m). Native resolution.

        Each row: inputs (eq_*, input_*, n, m) + outputs (200 profile values).
        Model learns: given (equilibrium, n, m), predict δp_n,m(ψ_N).

        **kwargs are passed to load_per_mode_for_surge (e.g. filename for
        sdata_complex_v2.h5 on CFS batch trees).
        """
        scripts_m3dc1 = Path(__file__).resolve().parent.parent.parent / "scripts" / "m3dc1"
        if str(scripts_m3dc1) not in sys.path:
            sys.path.insert(0, str(scripts_m3dc1))
        from dataset_complex_v2 import load_per_mode_for_surge

        df, input_cols, output_cols = load_per_mode_for_surge(
            batch_dir,
            run_pattern=run_pattern,
            sparc_pattern=sparc_pattern,
            spectrum_field=spectrum_field,
            spectrum_time_idx=spectrum_time_idx,
            use_magnitude=use_magnitude,
            max_cases=max_cases,
            verbose=verbose,
            **kwargs,
        )
        dataset = cls()
        dataset.load_from_dataframe(
            df,
            input_columns=input_cols,
            output_columns=output_cols,
        )
        dataset.file_path = Path(batch_dir)
        return dataset

    @classmethod
    def from_sdata_complex_v2(
        cls,
        path: Union[str, Path],
        *,
        delta_p_key: Optional[str] = None,
        use_magnitude: bool = True,
        input_features: Optional[list[str]] = None,
        include_equilibrium_features: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "M3DC1Dataset":
        """
        Load M3DC1 dataset from a single sdata_complex_v2.h5 file.

        Parameters
        ----------
        path : str or Path
            Path to sdata_complex_v2.h5
        delta_p_key : str, optional
            Key name for delta p spectra. If None, auto-detect.
        use_magnitude : bool
            If True, use magnitude for complex spectrum
        input_features : list of str, optional
            Input columns to include. If None, uses default set.
        include_equilibrium_features : bool
            Whether to include eq_R0, eq_a, etc.
        metadata_path : str or Path, optional
            Metadata YAML for input/output overrides
        **kwargs
            Passed to convert_sdata_complex_v2_to_dataframe

        Returns
        -------
        M3DC1Dataset
            Loaded dataset
        """
        _, convert_sdata_complex_v2_to_dataframe = _get_m3dc1_loaders()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"sdata_complex_v2 file not found: {path}")

        df = convert_sdata_complex_v2_to_dataframe(
            path,
            delta_p_key=delta_p_key,
            use_magnitude=use_magnitude,
            input_features=input_features,
            include_equilibrium_features=include_equilibrium_features,
            **kwargs,
        )
        input_cols = [
            c for c in df.columns
            if (c.startswith("eq_") and c != "eq_id")
            or c.startswith("input_")
            or c.startswith("eigenmode_amp_")
            or c in ("q0", "q95", "qmin", "p0")
        ]
        output_cols = [c for c in df.columns if c.startswith("output_p_")]

        dataset = cls()
        if metadata_path:
            dataset.metadata = dataset._load_metadata(metadata_path)
        dataset.load_from_dataframe(
            df,
            input_columns=input_cols,
            output_columns=output_cols,
        )
        dataset.file_path = path
        return dataset
