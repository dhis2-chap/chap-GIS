# Rwanda Land Cover — DLUP Fetch

## Requirements & installation

The script requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync
```

This installs all dependencies (currently just `requests`) into a local virtualenv.
Always run the script via `uv run` so it uses that environment:

```bash
uv run raw_scripts/rwanda_district_land_use_plan/fetch_dlup.py <input.geojson> --district <district>
```

Fetches Rwanda National Land Authority (NLA) District Land Use Plan (DLUP)
data for any area defined by an input GeoJSON polygon.

**Source:** Rwanda NLA ArcGIS FeatureServer  
**Layer:** DLUP land use zones

## Example

`ngoma_box_small.geojson` is a small bounding box (~0.5 km²) in Ngoma District,
suitable for a quick test run:

```bash
uv run raw_scripts/rwanda_district_land_use_plan/fetch_dlup.py \
    raw_scripts/rwanda_district_land_use_plan/ngoma_box_small.geojson \
    --district ngoma
```

**Output file** (written to `raw_scripts/rwanda_district_land_use_plan/outputs/`, created automatically):

| File | Contents |
|------|----------|
| `raw_scripts/rwanda_district_land_use_plan/outputs/ngoma_box_small_dlup.geojson` | DLUP land use zones intersecting the polygon |

Output file name is derived from the input stem: `raw_scripts/rwanda_district_land_use_plan/outputs/<stem>_dlup.geojson`.
Use `--out` to override the path entirely.

## Usage

```bash
uv run raw_scripts/rwanda_district_land_use_plan/fetch_dlup.py <input.geojson> --district <district> [--out output.geojson]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--district` | *(required)* | Rwanda district name (case-insensitive) |
| `--out` | auto | Output path; auto-derived from input stem if omitted |

The `--district` argument selects the correct NLA FeatureServer for that district.
All 30 Rwanda districts are supported (Gasabo, Kicukiro, and Nyarugenge share the
`Kigali_DLUP` service).

## Output fields

| Field | Description |
|-------|-------------|
| `zone_code` | Zone type code (e.g. `R1`, `W1A`, `A2`) |
| `zoning` | Full zoning description |
| `gen_lu` | General land use category |
| `area_ha` | Zone area in hectares |
| `phasing` | Development phasing |
| `legend_category` | Broad category (Agriculture, Wetland, Residential, …) |
| `legend_label` | `<code>-<full label>` string |

## Input files

| File | Description |
|------|-------------|
| `raw_scripts/rwanda_district_land_use_plan/ngoma_box_small.geojson` | Small test polygon in Ngoma District (~0.5 km²) |
| `raw_scripts/rwanda_district_land_use_plan/ngoma_box.geojson` | Larger Ngoma area polygon |
