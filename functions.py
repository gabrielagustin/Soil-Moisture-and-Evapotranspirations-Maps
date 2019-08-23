# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Mon Sep 10 14:37:54 2018

@author: gag

Necessary functions for opening and creating .HDF files. And to make the match between two .HDF files. 

"""


import numpy as np
from osgeo import gdal, ogr, gdalconst
import sys
import matplotlib.pyplot as plt


def openFileHDF(file, nroBand):
    """ 
    Function that opens an image with .HDF format and reads a specific band.

    Parameters:
    -----------
    file : complete path of the raster image 
    nroBand : number of the band to be read 

    Returns: 
    --------
    src_ds: source raster object
    band:
    GeoT: georeference
    Project: projection
    """

    try:
        src_ds = gdal.Open(file)
    except (RuntimeError, e):
        print('Unable to open File')
        print(e)
        sys.exit(1)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    #print cols
    #print rows
    bands = src_ds.RasterCount
    #print bands

    # se obtienen las caracteristicas de las imagen HDR
    GeoT = src_ds.GetGeoTransform()
    #print GeoT
    Project = src_ds.GetProjection()

    try:
        srcband = src_ds.GetRasterBand(nroBand)
    except(RuntimeError, e):
        # for example, try GetRasterBand(10)
        print('Band ( %i ) not found' % band_num)
        print(e)
        sys.exit(1)
    band = srcband.ReadAsArray()
    return src_ds, band, GeoT, Project


def createHDFfile(path, nameFileOut, driver, GeoT, Projection, img, xsize, ysize):
    """ 
    Function that creates a new .HDF file.

    Parameters:
    -----------
    path, nameFileOut: path and name of the file to be created,
    driver: file type
    GeoT, Projection: geotransform and projection data
    img: data that represent the image
    xsize, ysize: image size

    Returns: 
    --------

    """

    print("archivo creado:" + str(nameFileOut))
    driver = gdal.GetDriverByName(driver)
    ds = driver.Create(path + nameFileOut, xsize, ysize, 1, gdal.GDT_Float64)
    ds.SetProjection(Projection)
    geotransform = GeoT
    ds.SetGeoTransform(geotransform)
    ds.GetRasterBand(1).WriteArray(np.array(img))
    return


def matchData(data_src, data_match, nRow, nCol, type):
    """ 
    Function that performs the match to a raster data from a source raster modifying the projection,
    the transformation and the size. Different interpolation methods are used. 

    Parameters:
    -----------
    data_src: raster source
    data_match: raster to match
    nRow, nCol: raster size
    type: interpolation method ('Nearest', 'Bilinear', 'Cubic', 'Average')

    Returns: 
    --------
    data_result: a new raster created in memory
    """

    #data_result = gdal.GetDriverByName('MEM').Create('', data_match.RasterXSize, data_match.RasterYSize, 1, gdalconst.GDT_Float64)

    data_result = gdal.GetDriverByName('MEM').Create('', nCol, nRow, 1, gdalconst.GDT_Float64)

    # Se establece el tipo de proyección y transfomcion en resultado  que va ser coincidente con data_match
    data_result.SetGeoTransform(data_match.GetGeoTransform())
    data_result.SetProjection(data_match.GetProjection())

    # se cambia la proyeccion de data_src, con los datos de data_match y se guarda en data_result
    if (type == "Nearest"):
        gdal.ReprojectImage(data_src,data_result,data_src.GetProjection(),data_match.GetProjection(), gdalconst.GRA_NearestNeighbour)
    if (type == "Bilinear"):
        gdal.ReprojectImage(data_src, data_result, data_src.GetProjection(), data_match.GetProjection(), gdalconst.GRA_Bilinear)
    if (type == "Cubic"):
        gdal.ReprojectImage(data_src, data_result, data_src.GetProjection(), data_match.GetProjection(), gdalconst.GRA_Cubic)
    if (type == "Average"):
        gdal.ReprojectImage(data_src, data_result, data_src.GetProjection(), data_match.GetProjection(), gdal.GRA_Average)
    return data_result






