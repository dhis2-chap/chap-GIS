"""Download SRTM elevation for a country.

If a pre-existing elevation file is provided via --source, copies it.
Otherwise attempts to download via R/geodata using --project-root to
locate the R scripts.
"""

from pathlib import Path

import cyclopts

from malaria_research.utils.constants import COUNTRY_CONFIGS

app = cyclopts.App()


@app.default
def main(
    country: str = "rwanda",
    output: Path = Path("elevation.tif"),
    source: Path | None = None,
    project_root: Path | None = None,
) -> None:
    import shutil

    config = COUNTRY_CONFIGS[country.lower()]
    name = config.name.lower()

    output.parent.mkdir(parents=True, exist_ok=True)

    # If a source file is provided, just copy it
    if source is not None and source.exists():
        shutil.copy2(source, output)
        print(f"Copied: {source} -> {output}")
        return

    # Otherwise try to download via R
    import subprocess

    if project_root is None:
        project_root = Path.cwd()

    r_script = project_root / "scripts" / f"download_{name}_elevation.R"
    expected = project_root / "data" / "cache" / f"{name}_elevation.tif"

    if expected.exists():
        shutil.copy2(expected, output)
        print(f"Copied from cache: {expected} -> {output}")
        return

    if not r_script.exists():
        raise FileNotFoundError(f"R script not found: {r_script}")

    result = subprocess.run(
        ["Rscript", str(r_script)],
        capture_output=True, text=True, timeout=600,
        cwd=str(project_root),
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed: {result.stderr[-500:]}")

    if expected.exists():
        shutil.copy2(expected, output)
    else:
        raise FileNotFoundError(f"Expected output not created: {expected}")

    print(f"Saved: {output}")


if __name__ == "__main__":
    app()
