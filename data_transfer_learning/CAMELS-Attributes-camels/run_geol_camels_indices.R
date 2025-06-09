# library(sf)
# library(terra)

# print("Success 1")

# # Load your basin or shapefile
# basin <- st_read('/Users/brendansinclair/Desktop/ESDL Research/maps/RussianSubbasinsSimple.shp')

# print("Success2")

# basin <- st_transform(basin, crs = 4326)

# print("Success3")

# # Load GLHYMPS and GLiM GDB layers using sf
# glhymps <- st_read(dsn = "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-camels/dataverse_files/GLHYMPS", layer = "GLHYMPS")
# print("Success4")
# glim <- st_read(dsn = "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-camels/dataverse_files/LiMW_GIS_2015.gdb", layer = "GLiM_export")

# print("Success5")

# # Reproject to WGS84 if needed
# glhymps <- st_transform(glhymps, crs = 4326)

# print("Success6")

# glim <- st_transform(glim, crs = 4326)

# print("Success7")


# # Crop (spatial filter)
# glhymps_crop <- st_intersection(glhymps, basin)

# print("Success8")

# glim_crop <- st_intersection(glim, basin)

# # Save if desired
# saveRDS(glhymps_crop, "glhymps_hopland.rds")
# saveRDS(glim_crop, "glim_hopland.rds")

library(sf)
library(terra)

print("Success 1")

# Load your basin shapefile and convert CRS to WGS84
basin <- st_read('/Users/brendansinclair/Desktop/ESDL Research/maps/RussianSubbasinsSimple.shp')
print("Success2")
basin <- st_transform(basin, crs = 4326)
print("Success3")

# Load GLHYMPS shapefile and reproject
glhymps <- st_read(
  dsn = "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-camels/dataverse_files/GLHYMPS",
  layer = "GLHYMPS"
)
print("Success4")
glhymps <- st_transform(glhymps, crs = 4326)
print("Success5")

# Load GLiM from geodatabase (do not transform yet!)
glim <- st_read(
  dsn = "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-camels/dataverse_files/LiMW_GIS_2015.gdb",
  layer = "GLiM_export"
)
print("Success6")

# Crop first to reduce size, then transform
glim_crop <- st_crop(glim, basin)           # quick bounding box crop
print("Success7")
glim_crop <- st_make_valid(glim_crop)       # fix invalid geometries (optional)
glim_crop <- st_transform(glim_crop, 4326)  # now safe to transform
print("Success8")

# Intersect GLHYMPS with basin
glhymps_crop <- st_crop(glhymps, basin)
print("Success9")

# Save outputs
saveRDS(glhymps_crop, "glhymps_hopland.rds")
saveRDS(glim_crop, "glim_hopland.rds")
print("✅ All done!")
