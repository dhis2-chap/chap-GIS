"""Rice field Africa (20m) loader — Jiang et al. 2023, Zenodo record 13729353.

The dataset is hosted on Zenodo (CC-BY-4.0). ``download()`` fetches the
country's TIFF(s) automatically; tiled countries are merged on the fly into a
single ``data/inputs/{iso3}_rice_fields.tif``. If the dataset doesn't cover
the requested country, or auto-download fails, ``download()`` raises with the
manual instructions from the README.
"""

# NOTE: For alternative rice datasets and regions, see:
# https://www.sciencedirect.com/science/article/pii/S003442572600026X#bib508

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.parse
from pathlib import Path

import requests
import rioxarray
import xarray as xr
from geopandas import GeoDataFrame
from rioxarray.merge import merge_arrays

from .cache import cache_dir
from dhis2eo.utils.types import BBox, DateLike


logger = logging.getLogger(__name__)


dataset_id = "jiang_rice_fields"
ZENODO_RECORD = "13729353"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Zenodo file keys are English country names (or, for some, ISO3). This map
# resolves chap_gis ISO3 country codes to the Zenodo prefix used to find tiles.
# Africa-only — the dataset doesn't cover countries outside Africa.
_ISO3_TO_ZENODO_PREFIX = {
    "AGO": "Angola",
    "BEN": "Benin",
    "BFA": "Burkina Faso",
    "BDI": "Burundi",
    "CMR": "Cameroon",
    "CAF": "Central African Republic",
    "TCD": "Chad",
    "CIV": "CIV",
    "COD": "Democratic Republic of Congo",
    "EGY": "Egypt",
    "ETH": "Ethiopia",
    "GMB": "Gambia",
    "GHA": "Ghana",
    "GIN": "Guinea",
    "GNB": "Guinea-Bissau",
    "KEN": "Kenya",
    "LBR": "Liberia",
    "MDG": "Madagascar",
    "MWI": "Malawi",
    "MLI": "Mali",
    "MRT": "Mauritania",
    "MAR": "Morocco",
    "MOZ": "Mozambique",
    "NER": "Niger",
    "NGA": "Nigeria",
    "RWA": "Rwanda",
    "SEN": "Senegal",
    "SLE": "Sierra Leone",
    "SSD": "SouthSudan",
    "SDN": "Sudan",
    "TZA": "Tanzania",
    "TGO": "Togo",
    "UGA": "Uganda",
    "ZMB": "Zambia",
}

_README_INSTRUCTIONS = (
    "Manual fallback (also documented in README.md):\n"
    f"  1. Open https://zenodo.org/records/{ZENODO_RECORD}\n"
    "  2. Download the .tif file(s) for your country (Africa only).\n"
    "  3. If multiple tiles, merge them with `rioxarray.merge.merge_arrays`.\n"
    "  4. Save as `data/inputs/{iso3_lower}_rice_fields.tif`."
)


def _inputs_dir() -> Path:
    """Pre-staged inputs live alongside the cache, not inside it."""
    d = cache_dir().parent / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zenodo_files_for(prefix: str) -> list[dict]:
    """Return Zenodo file records whose key starts with ``prefix`` (with ``.tif``)."""
    response = requests.get(ZENODO_API, timeout=30)
    response.raise_for_status()
    record = response.json()
    matches: list[dict] = []
    for f in record.get("files", []):
        key = f["key"]
        if not key.endswith(".tif"):
            continue
        # Match either "Rwanda.tif" or "Chad-0000131072-0000000000.tif".
        stem = key[: -len(".tif")]
        if stem == prefix or stem.startswith(f"{prefix}-"):
            matches.append(f)
    return matches


def _download_file(url: str, dest: Path) -> None:
    logger.info(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            shutil.copyfileobj(response.raw, fh)


def _merge_into(tile_paths: list[Path], out_path: Path) -> None:
    arrays = [rioxarray.open_rasterio(p) for p in tile_paths]
    merged = merge_arrays(arrays)
    merged.rio.to_raster(out_path)


def download(
    start: DateLike | None = None,
    end: DateLike | None = None,
    bbox: BBox | None = None,
    *,
    dirname: str | Path | None = None,
    prefix: str | None = None,
    country_code: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Ensure the rice raster for ``country_code`` is staged under ``data/inputs/``.

    Auto-downloads from Zenodo (record 13729353, CC-BY-4.0). Tiled countries are
    merged into a single GeoTIFF. ``start``, ``end``, ``bbox`` are accepted for
    protocol symmetry but ignored.

    Raises :class:`ValueError` if the country isn't in the Africa dataset, and
    :class:`RuntimeError` (with the README fallback instructions) if the
    download fails for any other reason.
    """
    if not country_code:
        raise ValueError("rice.download requires country_code")
    iso3 = country_code.upper()
    if iso3 not in _ISO3_TO_ZENODO_PREFIX:
        raise ValueError(
            f"{iso3} is not in the Jiang et al. Africa rice dataset "
            f"(Zenodo {ZENODO_RECORD}).\n{_README_INSTRUCTIONS}"
        )

    save_path = _inputs_dir() / f"{iso3.lower()}_rice_fields.tif"
    if save_path.exists() and not overwrite:
        logger.info(f"Rice file already staged: {save_path}")
        return [save_path]

    zenodo_prefix = _ISO3_TO_ZENODO_PREFIX[iso3]
    try:
        files = _zenodo_files_for(zenodo_prefix)
        if not files:
            raise RuntimeError(
                f"No files matching '{zenodo_prefix}' on Zenodo {ZENODO_RECORD}."
            )
        with tempfile.TemporaryDirectory() as tmp:
            tile_paths = []
            for f in files:
                tmp_path = Path(tmp) / urllib.parse.unquote(f["key"])
                _download_file(f["links"]["self"], tmp_path)
                tile_paths.append(tmp_path)
            if len(tile_paths) == 1:
                shutil.move(tile_paths[0], save_path)
            else:
                logger.info(f"Merging {len(tile_paths)} tiles into {save_path}")
                _merge_into(tile_paths, save_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to auto-download rice fields for {iso3} from Zenodo "
            f"{ZENODO_RECORD}: {exc}\n{_README_INSTRUCTIONS}"
        ) from exc

    return [save_path]


def load(
    aoi: GeoDataFrame | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    country_code: str,
) -> xr.DataArray:
    """Load 20m Africa rice distribution (Jiang et al. 2023) for ``country_code``.

    Calls :func:`download` first to ensure the file is staged. ``aoi``,
    ``start``, ``end`` are accepted for protocol symmetry but ignored.
    """
    files = download(country_code=country_code)
    da = xr.open_dataarray(files[0])
    da = da.squeeze("band")

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "rice"
    da.attrs.update(
        long_name="Rice fields",
        standard_name="area_fraction",
        units="1",
        source="Jiang et al. 2023 — 20m Africa rice distribution map (Zenodo 13729353)",
    )
    return da
