#!/usr/bin/env python3
"""
fetch_dlup.py
─────────────
Fetches Rwanda National Land Authority (NLA) District Land Use Plan (DLUP)
data for any area defined by an input GeoJSON file.

Source: https://services7.arcgis.com/htgaiKX6RV2DDGgK/arcgis/rest/services/
Owner:  Rwanda National Land Authority (NLA)

Usage:
    python fetch_dlup.py <input.geojson> --district <district_name> [--out output.geojson]

Legend (all 46 zone types from the map):
    Agriculture:       A1, A2
    Buffer:            B1 (Wetland), B2 (Water body), B3 (Nat. park), B4 (Other)
    Commercial:        C1, C2, C3, C4
    Open Space:        ET (Eco-Tourism)
    Forest:            F1 (Plantation), F2 (National parks), F3 (NP expansion),
                       F4 (Natural), F5 (Afforestation)
    Industrial:        I1 (Light), I2 (General), I3 (Mining/Quarry)
    Public Admin:      PA
    Public Facility:   PF1 (Education), PF2 (Health), PF3 (Religious),
                       PF4 (Cultural), PF5 (Sport), PF6 (Cemetery)
    Residential:       R1, R1A, R1B, R2, R3, R4, RS
    Transportation:    T1 (Road), T2 (Bus), T3 (Airport), T4 (Railway)
    Utility:           U, UE
    Wetland:           W1A (Protected), W1B (Unprotected), W2 (Rehabilitation),
                       W3 (Sustainable), W4 (Conservation), W5 (Recreational)
    Water Body:        WB
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

# ─── NLA Service endpoints ────────────────────────────────────────────────────
BASE = "https://services7.arcgis.com/htgaiKX6RV2DDGgK/arcgis/rest/services"

# District → ArcGIS FeatureServer service name.
# Gisagara uses the "DLUPC" variant; Kigali covers Gasabo/Kicukiro/Nyarugenge.
DISTRICT_SERVICE = {
    "bugesera":   "Bugesera_DLUP",
    "burera":     "Burera_DLUP",
    "gakenke":    "Gakenke_DLUP",
    "gasabo":     "Kigali_DLUP",
    "gatsibo":    "Gatsibo_DLUP",
    "gicumbi":    "Gicumbi_DLUP",
    "gisagara":   "Gisagara_DLUPC",
    "huye":       "Huye_DLUP",
    "kamonyi":    "Kamonyi_DLUP",
    "karongi":    "Karongi_DLUP",
    "kayonza":    "Kayonza_DLUP",
    "kicukiro":   "Kigali_DLUP",
    "kirehe":     "Kirehe_DLUP",
    "muhanga":    "Muhanga_DLUP",
    "musanze":    "Musanze_DLUP",
    "ngoma":      "Ngoma_DLUP",
    "ngororero":  "Ngororero_DLUP",
    "nyabihu":    "Nyabihu_DLUP",
    "nyagatare":  "Nyagatare_DLUP",
    "nyamagabe":  "Nyamagabe_DLUP",
    "nyamasheke": "Nyamasheke_DLUP",
    "nyanza":     "Nyanza_DLUP",
    "nyarugenge": "Kigali_DLUP",
    "nyaruguru":  "Nyaruguru_DLUP",
    "rubavu":     "Rubavu_DLUP",
    "ruhango":    "Ruhango_DLUP",
    "rulindo":    "Rulindo_DLUP",
    "rusizi":     "Rusizi_DLUP",
    "rutsiro":    "Rutsiro_DLUP",
    "rwamagana":  "Rwamagana_DLUP",
}

BATCH_SIZE = 1000

# ─── Full legend (46 zone types) ─────────────────────────────────────────────
LEGEND = {
    # Agriculture
    "A1":  {"label": "Agriculture zone",                          "category": "Agriculture"},
    "A2":  {"label": "Livestock zone",                            "category": "Agriculture"},
    # Buffer
    "B1":  {"label": "Wetland buffer zone",                       "category": "Buffer"},
    "B2":  {"label": "Water body buffer zone",                    "category": "Buffer"},
    "B3":  {"label": "National park buffer zone",                 "category": "Buffer"},
    "B4":  {"label": "Other buffer zone",                         "category": "Buffer"},
    # Commercial
    "C1":  {"label": "Mixed use commercial zone",                 "category": "Commercial"},
    "C2":  {"label": "Neighbourhood commercial zone",             "category": "Commercial"},
    "C3":  {"label": "City commercial zone",                      "category": "Commercial"},
    "C4":  {"label": "Regional commercial zone",                  "category": "Commercial"},
    # Open Space
    "ET":  {"label": "Eco-Tourism and Open Space Zone",           "category": "Open Space"},
    # Forest
    "F1":  {"label": "Forest plantation zone",                    "category": "Forest"},
    "F2":  {"label": "National parks zone",                       "category": "Forest"},
    "F3":  {"label": "National park expansion zone",              "category": "Forest"},
    "F4":  {"label": "Natural forest zone",                       "category": "Forest"},
    "F5":  {"label": "Afforestation zone",                        "category": "Forest"},
    # Industrial
    "I1":  {"label": "Light industrial zone",                     "category": "Industrial"},
    "I2":  {"label": "General industrial zone",                   "category": "Industrial"},
    "I3":  {"label": "Mining, Extraction, Quarry",                "category": "Industrial"},
    # Public
    "PA":  {"label": "Public administration zone",                "category": "Public Administration"},
    "PF1": {"label": "Education and research facilities",         "category": "Public Facility"},
    "PF2": {"label": "Health facilities",                         "category": "Public Facility"},
    "PF3": {"label": "Religious facilities",                      "category": "Public Facility"},
    "PF4": {"label": "Cultural, Heritage, Memorial sites",        "category": "Public Facility"},
    "PF5": {"label": "Sport, Leisure facilities",                 "category": "Public Facility"},
    "PF6": {"label": "Cemetery, Crematoria",                      "category": "Public Facility"},
    # Residential
    "R1":  {"label": "Low density residential zone",              "category": "Residential"},
    "R1A": {"label": "Low density residential densification zone","category": "Residential"},
    "R1B": {"label": "Rural residential zone",                    "category": "Residential"},
    "R2":  {"label": "Medium density residential - Improvement",  "category": "Residential"},
    "R3":  {"label": "Medium density residential - Expansion",    "category": "Residential"},
    "R4":  {"label": "High density residential zone",             "category": "Residential"},
    "RS":  {"label": "Rural settlement site",                     "category": "Residential"},
    # Transportation
    "T1":  {"label": "Road reserve",                              "category": "Transportation"},
    "T2":  {"label": "Bus terminals and stations",                "category": "Transportation"},
    "T3":  {"label": "Airports, Ports and Connected facilities",  "category": "Transportation"},
    "T4":  {"label": "Railway and stations",                      "category": "Transportation"},
    # Utility
    "U":   {"label": "Utility zone",                              "category": "Utility"},
    "UE":  {"label": "Urban area extension zone",                 "category": "Utility"},
    # Wetland
    "W1A": {"label": "Wetland - Protected zone",                  "category": "Wetland"},
    "W1B": {"label": "Wetland - Unprotected zone",                "category": "Wetland"},
    "W2":  {"label": "Wetland - Rehabilitation zone",             "category": "Wetland"},
    "W3":  {"label": "Wetland - Sustainable exploitation zone",   "category": "Wetland"},
    "W4":  {"label": "Wetland - Conservation zone",               "category": "Wetland"},
    "W5":  {"label": "Wetland - Recreational zone",               "category": "Wetland"},
    # Water Body
    "WB":  {"label": "Waterbody zone",                            "category": "Water Body"},
}


def clip_features(features: list, aoi: dict) -> list:
    """Clip feature geometries to the AOI polygon."""
    aoi_shape = unary_union(
        [shape(f["geometry"]) for f in aoi.get("features", [])]
        if aoi.get("type") == "FeatureCollection"
        else [shape(aoi)]
    )
    clipped = []
    for f in features:
        geom = shape(f["geometry"]).intersection(aoi_shape)
        if geom.is_empty:
            continue
        f = {**f, "geometry": mapping(geom)}
        clipped.append(f)
    return clipped


def geojson_to_esri_envelope(geojson: dict) -> dict:
    """Compute bounding box of any GeoJSON (Feature, FeatureCollection, Geometry)."""
    coords = []

    def extract_coords(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                coords.append(obj[:2])
            else:
                for item in obj:
                    extract_coords(item)
        elif isinstance(obj, dict):
            if obj.get("type") == "FeatureCollection":
                for f in obj.get("features", []):
                    extract_coords(f)
            elif obj.get("type") == "Feature":
                extract_coords(obj.get("geometry", {}))
            elif "coordinates" in obj:
                extract_coords(obj["coordinates"])

    extract_coords(geojson)

    if not coords:
        raise ValueError("No coordinates found in GeoJSON")

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "xmin": min(lons), "ymin": min(lats),
        "xmax": max(lons), "ymax": max(lats),
        "spatialReference": {"wkid": 4326}
    }


def geojson_to_esri_geometry(geojson: dict) -> dict | None:
    """
    Convert a GeoJSON polygon/multipolygon to ESRI geometry for spatial filtering.
    Returns None if the input is not a polygon type.
    """
    def get_geometry(obj):
        if obj.get("type") == "FeatureCollection":
            geoms = [get_geometry(f) for f in obj.get("features", [])]
            geoms = [g for g in geoms if g]
            if not geoms:
                return None
            # Merge all rings into one multipolygon
            all_rings = []
            for g in geoms:
                all_rings.extend(g.get("rings", []))
            return {"rings": all_rings, "spatialReference": {"wkid": 4326}}
        elif obj.get("type") == "Feature":
            return get_geometry(obj.get("geometry", {}))
        elif obj.get("type") == "Polygon":
            return {"rings": obj["coordinates"], "spatialReference": {"wkid": 4326}}
        elif obj.get("type") == "MultiPolygon":
            rings = [ring for poly in obj["coordinates"] for ring in poly]
            return {"rings": rings, "spatialReference": {"wkid": 4326}}
        return None

    return get_geometry(geojson)


def fetch_layer(url: str, geometry: dict | None, envelope: dict,
                out_fields: str, label: str) -> list:
    """Paginated fetch of features within the given geometry/envelope."""
    all_features = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": BATCH_SIZE,
        }

        # Prefer exact polygon geometry if available, fall back to envelope
        if geometry:
            params["geometry"] = json.dumps(geometry)
            params["geometryType"] = "esriGeometryPolygon"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = "4326"
        else:
            params["geometry"] = json.dumps(envelope)
            params["geometryType"] = "esriGeometryEnvelope"
            params["spatialRel"] = "esriSpatialRelIntersects"
            params["inSR"] = "4326"

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ⚠ Request failed at offset {offset}: {e}", file=sys.stderr)
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        print(f"  [{label}] fetched {len(all_features)} features...", file=sys.stderr)

        if len(features) < BATCH_SIZE:
            break
        offset += BATCH_SIZE
        time.sleep(0.25)

    return all_features


def enrich_dlup(features: list) -> list:
    """Add legend metadata (category, full label) to DLUP features."""
    for f in features:
        props = f.get("properties", {})
        code = props.get("zone_code", "")
        if code in LEGEND:
            props["legend_category"] = LEGEND[code]["category"]
            props["legend_label"] = f"{code}-{LEGEND[code]['label']}"
        else:
            props["legend_category"] = props.get("gen_lu", "Unknown")
            props["legend_label"] = props.get("zoning", code)
    return features


def print_summary(features: list):
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"count": 0, "area_ha": 0.0})

    for f in features:
        p = f["properties"]
        cat = p.get("legend_category", "Unknown")
        by_cat[cat]["count"] += 1
        by_cat[cat]["area_ha"] += p.get("area_ha") or 0
    total_ha = sum(v["area_ha"] for v in by_cat.values())
    print(f"\n  {'Category':<25} {'Zones':>6}  {'Area (ha)':>10}  {'%':>5}")
    print(f"  {'-'*53}")
    for cat, s in sorted(by_cat.items(), key=lambda x: -x[1]["area_ha"]):
        pct = s["area_ha"] / total_ha * 100 if total_ha else 0
        print(f"  {cat:<25} {s['count']:>6}  {s['area_ha']:>10.1f}  {pct:>4.1f}%")
    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<25} {sum(v['count'] for v in by_cat.values()):>6}  {total_ha:>10.1f}")


def save_geojson(features: list, path: str, name: str):
    gj = {
        "type": "FeatureCollection",
        "name": name,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    with open(path, "w") as f:
        json.dump(gj, f, indent=2)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved → {path}  ({len(features)} features, {size_mb:.1f} MB)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NLA Rwanda DLUP data for a GeoJSON area of interest."
    )
    parser.add_argument("input", help="Input GeoJSON file (polygon area of interest)")
    parser.add_argument(
        "--district", required=True,
        help=f"Rwanda district name (case-insensitive). Supported: {', '.join(sorted(DISTRICT_SERVICE))}"
    )
    parser.add_argument("--out", default=None,
        help="Output file path (default: <input>_dlup.geojson)")
    args = parser.parse_args()

    district_key = args.district.lower()
    if district_key not in DISTRICT_SERVICE:
        print(
            f"Error: unknown district '{args.district}'.\n"
            f"Supported districts: {', '.join(sorted(DISTRICT_SERVICE))}",
            file=sys.stderr,
        )
        sys.exit(1)
    dlup_url = f"{BASE}/{DISTRICT_SERVICE[district_key]}/FeatureServer/0/query"

    # ── Load input GeoJSON ────────────────────────────────────────────────────
    with open(args.input) as f:
        aoi = json.load(f)

    print(f"\nInput:    {args.input}", file=sys.stderr)
    print(f"District: {args.district}  →  {DISTRICT_SERVICE[district_key]}", file=sys.stderr)

    try:
        envelope = geojson_to_esri_envelope(aoi)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    geometry = geojson_to_esri_geometry(aoi)

    print(f"Bounding box: {envelope['xmin']:.4f},{envelope['ymin']:.4f} → "
          f"{envelope['xmax']:.4f},{envelope['ymax']:.4f}", file=sys.stderr)
    print(f"Spatial filter: {'polygon' if geometry else 'envelope'}", file=sys.stderr)

    stem = Path(args.input).stem

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # ── Fetch DLUP zones ──────────────────────────────────────────────────────
    print("\nFetching DLUP land use zones...", file=sys.stderr)
    features = fetch_layer(
        dlup_url, geometry, envelope,
        out_fields="objectid,zone_code,zoning,gen_lu,landusecat,landuseold,area_ha,phasing",
        label="DLUP"
    )
    features = enrich_dlup(features)
    features = clip_features(features, aoi)
    out_path = args.out or str(out_dir / f"{stem}_dlup.geojson")
    save_geojson(features, out_path, f"{stem}_DLUP")
    print("\nDLUP summary:", file=sys.stderr)
    print_summary(features)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
