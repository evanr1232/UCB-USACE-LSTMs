# Load function scripts
source(paste(dir_r_scripts,'camels/clim/clim_indices.R',sep='')) # for month2sea
dir_r_scripts <- "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-"  # My code: assumes you're running from CAMELS-Attributes folder

source(paste(dir_r_scripts,'camels/hydro/hydro_signatures.R',sep='')) # for month2sea
dir_r_scripts <- "/Users/brendansinclair/Desktop/ESDL Research/UCB-USACE-LSTMs/data_transfer_learning/CAMELS-Attributes-"  # My code: assumes you're running from CAMELS-Attributes folder

source(paste(dir_r_scripts,'camels/time/time_tools.R',sep='')) # for month2sea

# Load your data for one basin (example: Hopland)
daymet <- read.csv("data_transfer_learning/CAMELS_US/basin_mean_forcing/daymet/spatial_avg_Hopland.csv")
# streamflow <- read.csv("../CAMELS_US/usgs_streamflow/11467000_streamflow_qc.txt")  # update filename
streamflow <- read.table(
  "data_transfer_learning/CAMELS_US/usgs_streamflow/11467000_streamflow_qc.txt",
  header = FALSE,      # ⬅️ No column names in your file
  fill = TRUE,         # ⬅️ Handles short lines gracefully
  col.names = c("gauge_id", "date", "flow", "quality")
)

streamflow$date <- as.Date(streamflow$date)
streamflow$flow <- as.numeric(streamflow$flow)

# Format date
daymet$Date <- as.Date(daymet$time)

# Create a complete daily date range
full_dates <- seq(min(daymet$Date), max(daymet$Date), by = "day")

# Merge to enforce a continuous daily time series
daymet_full <- merge(data.frame(Date = full_dates), daymet, by.x = "Date", by.y = "Date", all.x = TRUE)

# Interpolate missing values if any
library(zoo)
daymet_full$tmax <- na.approx(daymet_full$tmax, x = daymet_full$Date, rule = 1)
daymet_full$tmin <- na.approx(daymet_full$tmin, x = daymet_full$Date, rule = 1)
daymet_full$prcp <- na.approx(daymet_full$prcp, x = daymet_full$Date, rule = 1)

# Define inputs again using the cleaned data
temp <- (daymet_full$tmax + daymet_full$tmin) / 2
prec <- daymet_full$prcp
pet  <- rep(2, length(prec))  # still using dummy
day  <- daymet_full$Date
q    <- streamflow$flow[0:length(pet)]    # Select the first 15341 rows to match daymet data

# Compute CAMELS-style climatic indices
climate_indices <- compute_clim_indices_camels(temp, prec, pet, day, tol = 0.05)
print("Climatic Indices:")
print(climate_indices)

# Compute hydrologic signatures
hydro_signatures <- compute_hydro_signatures_camels(q, prec, day, tol = 0.05, hy_cal = "oct_us_gb")
print("Hydrologic Signatures:")
print(hydro_signatures)

