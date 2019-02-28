# Soil-Moisture-and-Evapotranspirations-Maps



Description: 

Generates maps of SM or ET from satellite images and the application of different models. 
The application of the model is basically performing matrix operations. Previously, satellite 
images must be modified to obtain the same projections and spatial resolutions.

Spatial resolutions are changed using reprojection operations based on interpolations of the type:

    -Nearest Neighbour
    -Bilinear
    -Cubic
    -Average

are used.



...


Dependences:

    python - sklearn
    python - mpl_toolkits
    python - skimage
    python - NumPy
    python - Matplolib
    python - Gdal



