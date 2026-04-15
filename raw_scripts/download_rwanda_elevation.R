#!/usr/bin/env Rscript
# Download SRTM ~90m elevation for all of Rwanda via geodata package.
# Saves to data/cache/rwanda_elevation.tif
#
# Rwanda bbox: ~28.86-30.90E, ~2.84S-1.05S
# elevation_3s downloads 5x5° tiles. Rwanda straddles the 30E boundary,
# so we need tiles from both the 25-30E and 30-35E columns.

library(terra)
library(geodata)

cache_dir <- "data/cache"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(cache_dir, "rwanda_elevation.tif")

if (file.exists(out_path)) {
  cat("Already cached:", out_path, "\n")
  quit(status = 0)
}

cat("Downloading SRTM elevation for Rwanda...\n")

# Download two tiles: west (25-30E) and east (30-35E)
elev_w <- elevation_3s(lon = 29.0, lat = -1.9, path = tempdir())
elev_e <- elevation_3s(lon = 30.5, lat = -1.9, path = tempdir())

# Merge and crop to Rwanda extent (with small buffer)
elev <- merge(elev_w, elev_e)
rwa_ext <- ext(28.8, 31.0, -2.9, -1.0)
elev <- crop(elev, rwa_ext)

writeRaster(elev, out_path, overwrite = TRUE)
cat("Saved:", out_path, "\n")
cat("Dimensions:", nrow(elev), "x", ncol(elev), "\n")
cat("Resolution:", res(elev), "\n")
cat("Extent:", as.vector(ext(elev)), "\n")
