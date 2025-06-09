Readme data file from Huscroft et al XX

When using the GLHYMPS 2.0 map data, please cite:
Huscroft et al. XX
When using the GLUM map data, please cite:

When using the original permeability compilation, please cite:
Gleeson, T., Smith. L., Jansen, N., Hartmann, J., Dürr, H., Manning, A.H.,van Beek, R. and A.M. Jellinek (2011) Mapping permeability over the surface of the earth. Geophysical Research Letters, 38, L02401, doi:10.1029/2010GL045565.

********Columns in shapefile******************************

*Indicates column can’t be deleted as it is locked. 
OBJECTID_1* - Master identifier for each polygon
SHAPE* - No value. Polygon listed in every row. 
IDENTITY_ - Original file name of polygon from original data source (ex. CHN33507) [data derived from GLHYMPS]
logK_Ice_x100_INT – Near surface global permeability values including permafrost areas (ex. logk = -2000). Values multiplied by 100 to become an integer. (This is the data to use if you want best representation of current global permeability with permafrost) In units m2
logK_Ferr_x100_INT – Near surface global permeability values including permafrost areas (ex. logk = -1090). Values multiplied by 100 to become an integer. (This is the data to use if you want best representation of current global permeability without permafrost) In units m2 	
Porosity_x100 – global porosity map with porosity multiplied by 100 so that it an integer. [data derived from GLHYMPS]
K_stdev_x100 – Standard deviation of logarithmic permeability values of logK_Ferr_x100_INT multiplied by 100. [data derived from GLHYMPS]
OBJECTID – Identifier used for GUM data
Description – Description of individual GUM polygon data taken from original data sources. (ex. Loess et lehms)
XX –Sediment sub-type. (ex.Yb = Beach deposit). ‘u’ values indicate undifferentiated and represent Sediment Type. [data derived from GUM]
YY– Grain size (ex. lc = silt/clay) [data derived from GUM]
ZZ – Mineralogy (ex. ac = acidic) [data derived from GUM]
AA – Age (ex. hu = Holocene) [data derived from GUM]
DD – Thickness in absolute values. ~12k values, don’t add much value [data derived from GUM]
Shape_Leng – Shape length provided by GUM. Does not match other shape_length columns [data derived from GUM]
GUM_K – 1 indicates values are derived from GLHYMPS; 2 indicates values are derived from GUM; 0 indicate areas absent permeability values 
Prmfrst – 1 indicates polygon within the permafrost boundary; 0 indicated polygon outside permafrost boundary [data derived from GUM]
Shape_Le_1 – Shape length added after re-projection [data derived from GUM]
Shape_Area – Shape area added after re-projection [data derived from GUM]
Transmissi – Transmissivity calculated from the multiplication of logK_Ferr_x100_INT and DTB_MEAN. In units m3/s [data derived from GUM]
COUNT – Number of different DTB values underlying individual polygon within DTB map [data derived from GUM]
AREA_1 – Area calculated using zonal statistics of individual polygon within DTB map [data derived from GUM]
MEAN – Calculated mean of DTB values underlying individual polygon within DTB map in centimeters [data derived from GUM]
STD – Standard deviation of DTB values underlying individual polygon within DTB map in centimeters [data derived from GUM]
********HOW TO CONVERT logk********************************

In the paper we show log(k) or log permeability (Figure 3). What we distribute is log(k)*100 or log permeability*100 so that it is an integer. Therefore k = 10^(k/100)

From some purposes, hydraulic conductivity is more useful. In the following, be careful of difference between little k (permeabilty) and big K (hydraulic conductivity).

K = k*rho*g/mu

Where K (m/s) is hydraulic conductivity which is dependent on fluid viscosity and density
rho (kg/m3) is the density of the fluid, normally water = 999.97 kg/m3
g (m/s2) is the acceleration due to gravity = 9.8 m/s2
mu (kg/m.s or Pa.s) is the viscosity of the fluid, normally water = 1e-3

Therefore, to convert permeability (m2) into hydaulic conductivity (m/s):

K = k*1e+7

Or if you want the full conversion of the mapped value: K = 10^(k/100)*1e+7
If you want to assume a 100 m thick aquifer, the transmissivity is therefore:
T = 10^(COLUMN/100)*1e+7*THICKNESS
 10^(k/100)*1e+7*100
