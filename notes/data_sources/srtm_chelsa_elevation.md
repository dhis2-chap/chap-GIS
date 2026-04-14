# SRTM / CHELSA Elevation & Climate

Topography and high-resolution climatology used for downscaling.

- **Variables:** Digital elevation model (SRTM); CHELSA monthly temperature climatology
- **Resolution:** 90 m (SRTM), ~1 km (CHELSA)
- **Coverage:** Global
- **Access:** R `geodata` package (SRTM via `elevation_3s`); CHELSA tiles downloaded as GeoTIFFs
- **Used in malaria-research:** `scripts/download_*_elevation.R`; lapse-rate temperature downscaling, exposure models
