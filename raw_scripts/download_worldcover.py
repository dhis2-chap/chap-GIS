"""Download ESA WorldCover tiles for a country."""

from pathlib import Path

import cyclopts

from malaria_research.utils.caching import download_file
from malaria_research.utils.constants import COUNTRY_CONFIGS, WC_BASE_URL

app = cyclopts.App()


@app.default
def main(country: str = "rwanda", output_dir: Path = Path("worldcover")) -> None:
    config = COUNTRY_CONFIGS[country.lower()]
    output_dir.mkdir(parents=True, exist_ok=True)
    for tile in config.worldcover_tiles:
        filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        url = f"{WC_BASE_URL}/{filename}"
        download_file(url, output_dir / filename, f"WorldCover {tile}")
    print(f"Downloaded {len(config.worldcover_tiles)} tiles to {output_dir}")


if __name__ == "__main__":
    app()
