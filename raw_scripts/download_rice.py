"""Download the Africa Rice Distribution Map 2023 (20m) for a country.

Source: Zenodo record 13729353 (per-country GeoTIFFs at ~20m resolution).
Only countries with ``has_rice_map=True`` in ``COUNTRY_CONFIGS`` are supported.
"""

from pathlib import Path

import cyclopts

from malaria_research.utils.caching import download_file
from malaria_research.utils.constants import COUNTRY_CONFIGS

RICE_BASE_URL = "https://zenodo.org/records/13729353/files"

app = cyclopts.App()


@app.default
def main(country: str = "rwanda", output_dir: Path = Path("rice")) -> None:
    config = COUNTRY_CONFIGS[country.lower()]
    if not config.has_rice_map:
        raise SystemExit(
            f"No Africa Rice Map 2023 tile available for {config.name}; "
            f"rice is absorbed into the wetland class for this country."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{config.name}.tif"
    url = f"{RICE_BASE_URL}/{filename}?download=1"
    local = output_dir / f"{config.iso3_lower}_rice_20m_2023.tif"
    download_file(url, local, f"{config.name} rice map 20m")
    print(f"Downloaded rice map to {local}")


if __name__ == "__main__":
    app()
