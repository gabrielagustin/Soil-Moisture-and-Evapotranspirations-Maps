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


Evapotranspiration maps examples:

- 36 km spatial resolucion (ET observada, ET modelada y Error):

<p align="center">
  <img width=285 src="2016_05_15_ETObservada.png"/>
  <img width=285 src="2016_05_15_ETmodelada.png"/>
  <img width=285 src="2016_05_15_Error.png"/>
</p>



Soil moisture maps examples:
- 30 m spatial resolucion (Multiple linear regression, Multilayer perceptron and Multivariate adaptive regression splines models):

<p align="center">
  <img width=285 src="2015-06-29_MLR.png"/>
  <img width=285 src="2015-06-29_MLP.png"/>
  <img width=285 src="2015-06-29_MARS.png"/>
</p>

...


Dependences:

    python - sklearn
    python - mpl_toolkits
    python - skimage
    python - NumPy
    python - Matplolib
    python - Gdal



