# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Mon Sep 10 14:37:54 2018

@author: gag
"""




import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python3/dist-packages/mpl_toolkits/')


from mpl_toolkits.basemap import Basemap, cm
import gdal
import matplotlib.pyplot as plt
import numpy.ma as np
from matplotlib import cm

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from osgeo import gdal, ogr
import sys
#from scipy.stats import threshold
from scipy import stats
import pandas as pd
import functions
import seaborn as sns


from numpy import linspace
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma





def normalizadoHR(c):
    min = 17.83
    max = 83.63
    new = (c -min)/(max-min)
    return new

def normalizadoTa(c):
    min = 6.9
    max = 26.29
    new = (c -min)/(max-min)
    return new


def normalizadoPP(c):
    min = 0
    max = 22.16
    new = (c -min)/(max-min)
    return new

def normalizadoSAR(c):
    min = -17.82
    max = -4.39
    new = (c -min)/(max-min)
    return new

def normalizado(c):
    min = np.min(c)
    max = np.max(c)
    new = (c -min)/(max-min)
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new


def normalizadoEt_etapa3(c):
    min = 1.28
    max = 25.39
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new

def normalizadoTs_etapa3(c):
    min = 7.45
    max = 26.5
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new


def normalizadoPP_etapa3(c):
    min = 0
    max = 139
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new
    
def normalizadoSAR_etapa3(c):
    min = -17.459
    max = -4.01
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new



def geospatial_coor(nameFile):
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    #Tvis-animated.gif
    try:
        src_ds = gdal.Open(nameFile)
    except(RuntimeError, e):
        print('Unable to open INPUT.tif')
        print(e)
        sys.exit(1)
    gt1 = src_ds.GetGeoTransform()
    ##### r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    #if "MODIS" in nameFile:
        #print "ACA MODIS"
        #print src_ds.RasterXSize
        #print src_ds.RasterYSize
        #print gt1
        #r1 = [gt1[0], gt1[3], gt1[3] + (gt1[5] * src_ds.RasterYSize), gt1[0] + (gt1[1] * src_ds.RasterXSize)]
    #else:
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * src_ds.RasterXSize), gt1[3] + (gt1[5] * src_ds.RasterYSize)]
    return r1

def openImage(nameFile,intersection):
        # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    band_num = 1
    #Tvis-animated.gif
    try:
        src_ds = gdal.Open(nameFile)
    except(RuntimeError, e):
        print('Unable to open INPUT.tif')
        print(e)
        sys.exit(1)
    try:
        srcband = src_ds.GetRasterBand(band_num)
    except(RuntimeError, e):
        # for example, try GetRasterBand(10)
        print('Band ( %i ) not found' % band_num)
        print(e)
        sys.exit(1)
    gt1 = src_ds.GetGeoTransform()
    Project = src_ds.GetProjection()
    print(nameFile)
    print(gt1)
    xOrigin = gt1[0]
    yOrigin = gt1[3]
    pixelWidth = float(gt1[1])
    pixelHeight = float(gt1[5])
    xmin = intersection[0]
    ymax = intersection[1]
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((np.abs(intersection[0])-np.abs(intersection[2]))/pixelWidth)
    ycount =  int((np.abs(intersection[1])-np.abs(intersection[3]))/pixelHeight)
    #print nameFile
    if (xoff == 0 and yoff == 0):
        lc = srcband.ReadAsArray()
    else:
        #print nameFile
        #print xoff
        #print yoff
        #print xcount
        #print ycount
        #if "MODIS" in nameFile:
            #lc = srcband.ReadAsArray(xoff, yoff, int(xcount), int(ycount))
        #else:
        lc = srcband.ReadAsArray(xoff, yoff, int(xcount), int(ycount))
    return src_ds, lc, gt1, Project

def applyNDVIfilter(sar,L8, etapa):
    result = sar
    rSar, cSar = sar.shape
    mask = np.ones((rSar, cSar))
    #print cFactor
    if (etapa == "etapa1"):
        print("NDVI filter")
        for i in range(0, rSar):
            for j in range(0, cSar):
                if (L8[i, j] > 0.51 ): mask[i,j] = 0
                if (L8[i, j] < 0.0 ): mask[i,j] = 0
    if (etapa == "etapa2" or etapa == "etapa3"):
        for i in range(0, rSar):
            for j in range(0, cSar):
                if (L8[i, j] > 0.8 ): mask[i,j] = 0
                if (L8[i, j] < 0.1 ): mask[i,j] = 0
    result = sar*mask
    return result,mask


def applyWaterfilter(sar,modis):
    #print sar.shape
    rSar, cSar = sar.shape
    mask = np.zeros((rSar, cSar))
    #print modis.shape
    rModis, cModis = modis.shape
    for i in range(0, rSar):
        for j in range(0, cSar):
            if (modis[i, j] < 0.1): mask[i,j] = 1
    #mask = np.ones((r,c))
    result = sar*mask*-30
    return result, mask



def applyBackfilter(sar):
    result = stats.threshold(sar, threshmin=-18, threshmax=-6, newval=0)
    print(result)
    #result = sgn.medfilt(result, 9)
    #result[result < 0] = 1
    mask = result
    mask[mask < 0] = 1
    result = result*mask

    r,c = sar.shape
    mask = np.ones((r,c))
    mask = mask - result
    return result, mask



def applyCityfilter(sar, L8_maskCity):
    r, c = sar.shape
    result = sar
    mask = np.ones((r,c))
    for i in range(0, r):
        for j in range(0, c):
            if (L8_maskCity[i, j] != 0 ): mask[i,j] = 0
    result = result*mask
    return result, mask



def meteoMap (sar, meteo):
    rSar, cSar = sar.shape
    sarMeteo = np.zeros((rSar, cSar))
    rMeteo, cMeteo = meteo.shape
    rFactor = rSar/float(rMeteo)
    #print rFactor
    cFactor = cSar/float(cMeteo)
    #print cFactor
    for i in range(0, rSar):
        for j in range(0, cSar):
            indexi = int(i/rFactor)
            indexj = int(j/cFactor)
            sarMeteo[i,j] = meteo[indexi, indexj]
    return sarMeteo



def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output



def calculateMaps(MLRmodel, MARSmodel, MLPmodel, etapa):
    print("-------------------------------------------------------------------")
    print("Calculate SM maps")
    #def calculateMaps(MLRmodel, MLPmodel, etapa):
    #dir = "ggarcia"
    dir = "gag"



    fechaSentinel = []
    fechaNDVI = []
    fechaLandsat8 =[]
    fechaSMAP =[]
    fechaMYD =[]

    if (etapa == "etapa1"):
        path = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Modelo/mapasCreados/Etapa1/"
        print(etapa)
        fechaSentinel.append("2015-06-29")
        fechaLandsat8.append("2015-06-18")
        fechaSMAP.append("2015-06-30")
        fechaSentinel.append("2015-10-03")
        fechaLandsat8.append("2015-10-08")
        fechaSMAP.append("2015-10-04")
        fechaSentinel.append("2015-12-28")
        fechaLandsat8.append("2015-12-27")
        fechaSMAP.append("2015-12-28")
        fechaSentinel.append("2016-03-19")
        fechaLandsat8.append("2016-03-16")
        fechaSMAP.append("2016-03-12")

        Ta = []
        HR = []
        PP = []
        sigma0 = []


        for i in range(0,len(fechaSentinel)):
            
            print("-----------------------------------------------------------")
            print(fechaSentinel[i])
            print("-----------------------------------------------------------")

            fileTa = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/T_aire.asc"
            src_ds_Ta, bandTa, GeoTTa, ProjectTa = functions.openFileHDF(fileTa, 1)

            filePP = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/PP.asc"
            src_ds_PP, bandPP, GeoTPP, ProjectPP = functions.openFileHDF(filePP, 1)

            fileHR = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Datos INTA/"+fechaSentinel[i]+"/HR.asc"
            src_ds_HR, bandHR, GeoTHR, ProjectHR = functions.openFileHDF(fileHR, 1)

                        
            fileNDVI = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Landsat8/"+fechaLandsat8[i]+"/NDVI_recortado"
            src_ds_NDVI, bandNDVI, GeoTNDVI, ProjectNDVI = functions.openFileHDF(fileNDVI, 1)


#            ##### smap a 10 km 
#            fileSMAP = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-10km/"+fechaSMAP[i]+"/soil_moisture.img"
#            print(fileSMAP)
#            src_ds_SMAP, bandSMAP, GeoTSMAP, ProjectSMAP = functions.openFileHDF(fileSMAP, 1)

            ##### CONAE interpolado
#            CONAE_HS = "/home/gag/Escritorio/inter_HS_29_06_2015.asc"
#            src_ds_CONAE_HS, bandCONAE_HS, GeoTCONAE_HS, ProjectCONAE_HS = functions.openFileHDF(CONAE_HS, 1)




            #fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel/"+fechaSentinel[i]+".SAFE/subset.data/recorte_30mx30m.img"
            #fileSar ="/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Sentinel-Otras/"+fechaSentinel[i]+".SAFE/subset.data/recorte_30mx30m.img"
            fileSar ="/media/"+dir+"/TOURO Mobile/Sentinel_30m_1km/"+fechaSentinel[i]+"/subset_30m_mapa.data/Sigma0_VV_db.img"


            nameFileMLR = "mapa_MLR_30m_"+str(fechaSentinel[i])
            nameFileMARS= "mapa_MARS_30m_"+str(fechaSentinel[i])
            nameFileMLP = "mapa_MLP_30m_"+str(fechaSentinel[i])


            src_ds_Sar, bandSar, GeoTSar, ProjectSar = functions.openFileHDF(fileSar, 1)
            print(ProjectSar)


            fileMascara = "/media/"+dir+"/Datos/Trabajos/Trabajo_Sentinel_NDVI_CONAE/Landsat8/2015-06-18/mascaraciudadyalgomas_reprojected/subset_1_of_Band_Math__b1_5.data/Band_Math__b1_5.img"
            src_ds_Mas, bandMas, GeoTMas, ProjectMas = functions.openFileHDF(fileMascara, 1)

            ### se cambian las resoluciones de todas las imagenes a la de la sar
            #type = "Nearest"
            type = "Bilinear"
            nRow, nCol = bandSar.shape

            data_src = src_ds_Mas
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchCity = match.ReadAsArray()

            data_src = src_ds_Ta
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchTa = match.ReadAsArray()

            print("------------------------------------------------------------")
            print("Max Ta: "+str(np.max(band_matchTa)))
            print("Min Ta: "+str(np.min(band_matchTa)))

            #fig, ax = plt.subplots()
            #ax.imshow(band_matchTa, interpolation='None',cmap=cm.gray)
            #plt.show()


            data_src = src_ds_PP
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchPP = match.ReadAsArray()

            print("Max PP: "+str(np.max(band_matchPP)))
            print("Min PP: "+str(np.min(band_matchPP)))

            #fig, ax = plt.subplots()
            #ax.imshow(band_matchPP, interpolation='None',cmap=cm.gray)
            #plt.show()

            data_src = src_ds_HR
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchHR = match.ReadAsArray()

            print("Max HR: "+str(np.max(band_matchHR)))
            print("Min HR: "+str(np.min(band_matchHR)))

            #HR = pd.DataFrame({'HR':band_matchHR.flatten()})
            #fig, ax = plt.subplots()
            #sns.distplot(HR)


            #print "------------------------------------------------------------"
            #fig, ax = plt.subplots()
            #ax.imshow(band_matchHR, interpolation='None',cmap=cm.gray)

            data_src = src_ds_NDVI
            data_match = src_ds_Sar
            match = functions.matchData(data_src, data_match, type, nRow, nCol)
            band_matchNDVI = match.ReadAsArray()
            
#            fig, ax = plt.subplots()
#            ax.imshow(band_matchNDVI, interpolation='None',cmap=cm.gray)
#            plt.show()
#            


            type = "Nearest"
#            data_src = src_ds_SMAP
#            data_match = src_ds_Sar
#            match = functions.matchData(data_src, data_match, type, nRow, nCol)
#            band_matchSMAP = match.ReadAsArray()


#            data_src = src_ds_CONAE_HS
#            data_match = src_ds_Sar
#            match = functions.matchData(data_src, data_match, type, nRow, nCol)
#            band_matchCONAE_HS = match.ReadAsArray()



            ### se filtra la imagen SAR
            #print "Se filtran las zonas con NDVI mayores a 0.51 y con NDVI menores a 0"
            sarEnmask, maskNDVI = applyNDVIfilter(bandSar, band_matchNDVI, etapa)
            
#            rSar, cSar = maskNDVI.shape
#            maskNDVI2 = np.zeros((rSar, cSar))
#            for i in range(0, rSar):
#                for j in range(0, cSar):
#                    if (maskNDVI[i, j] == 0 ): maskNDVI2[i,j] = 1


            filtWater, maskWater = applyWaterfilter(bandSar,band_matchNDVI)
            

            ### histograma de Sigma0 despues de filtrar
            #Ss = pd.DataFrame({'Sigma0':sarEnmask.flatten()})
            #fig, ax = plt.subplots()
            #sns.distplot(Ss)

#            sarEnmask, maskCity = applyCityfilter(sarEnmask,L8maskCity)
#            sarEnmask, maskSAR = applyBackfilter(sarEnmask)
            
            
            sarEnmask[sarEnmask < -18] = -0
            sarEnmask[sarEnmask > -4] = -4
 
            print("Max Sigma0: "+str(np.max(sarEnmask)))
            print("Min Sigma0: "+str(np.min(sarEnmask)))
            print("------------------------------------------------------------")
         

#            fig, ax = plt.subplots()
#            ax.imshow(sarEnmask, interpolation='None',cmap=cm.gray)
#            
#            fig, ax = plt.subplots()
#            ax.imshow(maskNDVI, interpolation='None',cmap=cm.gray)
#            plt.show()



            sarEnmask1 = np.copy(sarEnmask)
            sarEnmask2 = np.copy(sarEnmask)
            sarEnmask3 = np.copy(sarEnmask)

            r,c = bandSar.shape
            
            

#            OldRange = (np.max(band_matchPP)  - np.min(band_matchPP))
#            NewRange = (1 + 1)
#            newPP = (((band_matchPP - np.min(band_matchPP)) * NewRange) / OldRange) -1
#
#            OldRange = (np.max(sarEnmask1)  - np.min(sarEnmask1))
#            NewRange = (1 + 1)
#            sarEnmask1 = (((sarEnmask1 - np.min(sarEnmask1)) * NewRange) / OldRange) -1


            ### se normalizan las variables entre 0 y 1
            sarEnmask_22 = normalizadoSAR(sarEnmask1)
            PP_Norm = normalizadoPP(band_matchPP)
            Ta_Norm = normalizadoTa(band_matchTa)
            HR_Norm = normalizadoHR(band_matchHR)
            
            
            print("Max sarEnmask_22 norm:" +str(np.max(sarEnmask_22)))
            print("Min sarEnmask_22 norm:" +str(np.min(sarEnmask_22)))  
          
            print("Max PP norm:" +str(np.max(PP_Norm)))
            print("Min PP norm:" +str(np.min(PP_Norm)))
            
            print("Max Ta norm:" +str(np.max(Ta_Norm)))
            print("Min Ta norm:" +str(np.min(Ta_Norm)))
            
            
            print("Max HR norm:" +str(np.max(HR_Norm)))
            print("Min HR norm:" +str(np.min(HR_Norm)))
            
#            fig, ax = plt.subplots()
#            ax.imshow(PP_Norm, interpolation='None',cmap=cm.gray)
#
#            fig, ax = plt.subplots()
#            ax.imshow(np.log10(Ta_Norm), interpolation='None',cmap=cm.gray)
#
#
#            fig, ax = plt.subplots()
#            ax.imshow(np.log10(HR_Norm), interpolation='None',cmap=cm.gray)


            #### -------------------MLR method-------------------

            dataMap_MLR = pd.DataFrame({'Sigma0':sarEnmask_22.flatten(),'T_aire':(np.log10(Ta_Norm)).flatten(), 'HR':(np.log10(HR_Norm)).flatten(),'PP':PP_Norm.flatten()})
            dataMap_MLR = dataMap_MLR[[ 'T_aire', 'PP','Sigma0', 'HR']]
#            print(dataMap_MLR.describe())
#            input()
            dataMap_MLR = dataMap_MLR.fillna(0)
            mapSM_MLR = MLRmodel.predict(dataMap_MLR)
            ## debo invertir la funcion flatten()
            #mapSM_MLR = mapSM_MLR.reshape((r,c))
            mapSM_MLR = np.array(mapSM_MLR).reshape(r,c)
            mapSM_MLR = 10**(mapSM_MLR)
            mapSM_MLR[mapSM_MLR < 0] = 0
            #mapSM_MLR[mapSM_MLR > 60] = 0


            #### los datos para el modelo MLR llevan log
#            fig, ax = plt.subplots()
#            plt.hist(mapSM_MLR, bins=10)  # arguments are passed to np.histogram
#            plt.title("Histogram MLR maps")
#            plt.show()



            mapSM_MLR = mapSM_MLR*maskNDVI#*maskCity


            #SM = pd.DataFrame({'SM':mapSM_MLR.flatten()})
            #SM = SM[SM.SM != 0]
            #fig, ax = plt.subplots()
            #sns.distplot(SM)


            #fig, ax = plt.subplots()
            #ax.imshow(mapSM_MLR, interpolation='None',cmap=cm.gray)
            #plt.show()

            #plt.hist(mapSM_MLR)  # arguments are passed to np.histogram
            #plt.title("Histogram with 'auto' bins")
            #plt.show()

            print("MLR")
            print("Max:" +str(np.max(mapSM_MLR[np.nonzero(mapSM_MLR)])))
            print("Min:" +str(np.min(mapSM_MLR[np.nonzero(mapSM_MLR)])))
            print("Mean:" +str(np.mean(mapSM_MLR[np.nonzero(mapSM_MLR)])))
            print("STD:" +str(np.std(mapSM_MLR[np.nonzero(mapSM_MLR)])))

            #fig, ax = plt.subplots()
            #ax.imshow(mapSM_MLR, interpolation='None',cmap=cm.gray)


            #### -------------------MARS method-------------------

            dataMap_MARS = pd.DataFrame({'Sigma0' :sarEnmask.flatten(),'T_aire' :band_matchTa.flatten(), 'HR' :band_matchHR.flatten(),'PP' :band_matchPP.flatten()})
            dataMap_MARS = dataMap_MARS[[ 'T_aire', 'PP','Sigma0', 'HR']]
            dataMap_MARS = dataMap_MARS.fillna(0)
            mapSM_MARS = MARSmodel.predict(dataMap_MARS)
            ## debo invertir la funcion flatten()
            mapSM_MARS = mapSM_MARS.reshape(r,c)
            mapSM_MARS[mapSM_MARS < 0] = 0

            mapSM_MARS = mapSM_MARS*maskNDVI#*maskCity


            ####------------------- MLP method -------------------


#            OldRange = (np.max(band_matchTa)  - np.min(band_matchTa))
#            NewRange = (1 + 1)
#            Ta = (((band_matchTa - np.min(band_matchTa)) * NewRange) / OldRange) -1
#
#            OldRange = (np.max(band_matchHR)  - np.min(band_matchHR))
#            NewRange = (1 + 1)
#            HR = (((band_matchHR - np.min(band_matchHR)) * NewRange) / OldRange) -1
#
#            OldRange = (np.max(band_matchPP)  - np.min(band_matchPP))
#            NewRange = (1 + 1)
#            PP = (((band_matchPP - np.min(band_matchPP)) * NewRange) / OldRange) -1
#
#            OldRange = (np.max(sarEnmask)  - np.min(sarEnmask))
#            NewRange = (1 + 1)
#            sar2 = (((sarEnmask - np.min(sarEnmask)) * NewRange) / OldRange) -1


            OldRange = (26.29  - 6.9)
            NewRange = (1 + 1)
            Ta = (((band_matchTa - 6.9) * NewRange) / OldRange) -1

            OldRange = (83.63  - 17.83)
            NewRange = (1 + 1)
            HR = (((band_matchHR - 17.83) * NewRange) / OldRange) -1

            OldRange = (22.16  - 0)
            NewRange = (1 + 1)
            PP = (((band_matchPP - 0) * NewRange) / OldRange) -1

            OldRange = (-4.39  +17.82)
            NewRange = (1 + 1)
            sar2 = (((sarEnmask + 17.82) * NewRange) / OldRange) -1



            dataMap_MLP = pd.DataFrame({'T_aire' :Ta.flatten(),'Sigma0' :sar2.flatten(), 'HR' :HR.flatten(), 'PP' :PP.flatten()})
            dataMap_MLP = dataMap_MLP[[ 'T_aire', 'PP','Sigma0', 'HR']]

            #print dataMap_MLP
            ###.describe()
            dataMap_MLP = dataMap_MLP.fillna(0)
            mapSM_MLP = MLPmodel.predict(dataMap_MLP)
            mapSM_MLP = mapSM_MLP.reshape(r,c)
            #print mapSM_MLR.shape
            mapSM_MLP[mapSM_MLP < 0] = 0
            mapSM_MLP = mapSM_MLP*maskNDVI
            #fig, ax = plt.subplots()
            #ax.imshow(mapSM_MLP, interpolation='None',cmap=cm.gray)
            #plt.show()


            my_cmap = cm.Blues
            my_cmap.set_under('k', alpha=0)
            my_cmap1 = cm.Greens
            my_cmap1.set_under('k', alpha=0)
            my_cmap2 = cm.OrRd
            my_cmap2.set_under('k', alpha=0)
            my_cmap3 = cm.Oranges
            my_cmap3.set_under('k', alpha=0)

            transform = GeoTSar
            xmin,xmax,ymin,ymax=transform[0],transform[0]+transform[1]*src_ds_Sar.RasterXSize,transform[3]+transform[5]*src_ds_Sar.RasterYSize,transform[3]
            print(xmin)
            print(xmax)

            # plot MLR maps
            ##plt.hist(mapSM_MLR, bins=10)  # arguments are passed to np.histogram
            ##plt.title("Histogram with 'auto' bins")
            ##plt.show()

            fig, ax = plt.subplots()
#            meridians = [xmin, xmax,5]
            m = Basemap(projection='merc',llcrnrlat=ymin,urcrnrlat=ymax,\
            llcrnrlon=xmin,urcrnrlon=xmax,resolution='c')

#            m = Basemap(projection='cyl',llcrnrlat=ymin,urcrnrlat=ymax,\
#            llcrnrlon=xmin,urcrnrlon=xmax,resolution='c')

#            m.drawcoastlines()
#            lat = np.arange(xmin, xmax, 5)
#            print lat
#            m.drawlsmask(land_color='white',ocean_color='white',lakes=True)
#
#            m.drawparallels([-32.90,-32.95,-33.00,-33.05],labels=[1,0,0,0],fontsize=12, linewidth=0.0)
#
#            m.drawmeridians([-62.56,-62.48,-62.40,-62.32],labels=[0,0,1,0],fontsize=12, linewidth=0.0)
            
#            m.drawmapscale(-62.35, -33.04, 0,0, 10, barstyle='fancy', units='km')
            m.drawmapscale(-62.55, -33.05, 0,0, 10, fontsize = 10, units='km')
#            m.drawmapboundary(fill_color='aqua')
#            ax.add_compass(loc=1)
            
            sarEnmask[sarEnmask != -4] = 0
#            sarEnmask[sarEnmask == -4] = 1
            img = ax.imshow(sarEnmask, extent=[xmin,xmax,ymin,ymax], cmap=cm.gray, interpolation='none')
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.tick_top()
#            m.colorbar(img)
            
#            plt.title('Plot gridded data on a map')
            
            plt.savefig('gridded_data_global_map.png', pad_inches=0.5, bbox_inches='tight')
            ax.grid(False)
            plt.show()



            #im0 = ax.imshow(mapSM_MLR, cmap=my_cmap3)#, vmin=5, vmax=55, extent=[xmin,xmax,ymin,ymax], interpolation='None')
            #maskNDVI2 = ma.masked_where(maskNDVI2 == 0,maskNDVI2)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap1)




#            im0 = ax.imshow(maskNDVI, extent=[xmin,xmax,ymin,ymax],cmap=cm.gray)
#            im0.tick_labels.set_xformat('hhmm')
#            im0.tick_labels.set_yformat('hhmm')
#            
            plt.show()




            fig, ax = plt.subplots()

            #im0 = ax.imshow(mapSM_MLR, cmap=my_cmap3)#, vmin=5, vmax=55, extent=[xmin,xmax,ymin,ymax], interpolation='None')
            #maskNDVI2 = ma.masked_where(maskNDVI2 == 0,maskNDVI2)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap1)

            im0 = ax.imshow(mapSM_MLR, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap1, clim=(5, 45))

            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
            im=ax.imshow(pp, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap2, interpolation='Bilinear')
            kk = ma.masked_where(filtWater == 0, filtWater)
            im = ax.imshow(kk, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap, interpolation='Bilinear')
            ax.grid(False)
            ax.xaxis.tick_top()
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="5%", pad=0.05)
            cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
            cb.set_label('Volumetric SM (%)')
            #cb.set_clim(vmin=5, vmax=50)

            ### ----------------------------------------------------------------

            # plot MARS maps

            print("MARS")
            print("Max:" +str(np.max(mapSM_MARS[np.nonzero(mapSM_MARS)])))
            print("Min:" +str(np.min(mapSM_MARS[np.nonzero(mapSM_MARS)])))
            print("Mean:" +str(np.mean(mapSM_MARS[np.nonzero(mapSM_MARS)])))
            print("STD:" +str(np.std(mapSM_MARS[np.nonzero(mapSM_MARS)])))

            fig, ax = plt.subplots()
            im0 = ax.imshow(mapSM_MARS, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap1, clim=(5, 45))
            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
            im=ax.imshow(pp, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap2, interpolation='Bilinear')
            kk = ma.masked_where(filtWater == 0, filtWater)
            im = ax.imshow(kk, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap, interpolation='Bilinear')
            ax.grid(False)
            ax.xaxis.tick_top()
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="5%", pad=0.05)
            cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
            cb.set_label('Volumetric SM (%)')
            #cb.set_clim(vmin=5, vmax=50)

            ### ----------------------------------------------------------------

            # plot MLP map
            print("MLP")
            print("Max:" +str(np.max(mapSM_MLP[np.nonzero(mapSM_MLP)])))
            print("Min:" +str(np.min(mapSM_MLP[np.nonzero(mapSM_MLP)])))
            print("Mean:" +str(np.mean(mapSM_MLP[np.nonzero(mapSM_MLP)])))
            print("STD:" +str(np.std(mapSM_MLP[np.nonzero(mapSM_MLP)])))

            fig, ax = plt.subplots()
            im0= ax.imshow(mapSM_MLP, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap1, clim=(5, 45))
            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
            ax.imshow(pp, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap2, interpolation='Bilinear')
            kk = ma.masked_where(filtWater == 0, filtWater)
            ax.imshow(kk, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap, interpolation='Bilinear')
            ax.grid(False)
            ax.xaxis.tick_top()
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size="5%", pad=0.05)
            cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
            cb.set_label('Volumetric SM (%)')
            #cb.set_clim(vmin=5, vmax=50)


            ### ----------------------------------------------------------------

#            # plot SMAP map
#            
#            SMAP_SM = band_matchSMAP*100
#            SMAP_SM = maskNDVI*SMAP_SM
#            
#            print("SMAP SM")
#            print("Max:" +str(np.max(SMAP_SM[np.nonzero(SMAP_SM)])))
#            print("Min:" +str(np.min(SMAP_SM[np.nonzero(SMAP_SM)])))
#            print("Mean:" +str(np.mean(SMAP_SM[np.nonzero(SMAP_SM)])))
#            print("STD:" +str(np.std(SMAP_SM[np.nonzero(SMAP_SM)])))
#            
#
#            fig4, ax4 = plt.subplots()
#            im4= ax4.imshow(SMAP_SM, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap1, clim=(5, 45))
#            pp = ma.masked_where(band_matchCity == 0, band_matchCity)
#            ax4.imshow(pp, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap2, interpolation='Bilinear')
#            kk = ma.masked_where(filtWater == 0, filtWater)
#            ax4.imshow(kk, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap, interpolation='Bilinear')
#            ax4.grid(False)
#            ax4.xaxis.tick_top()
#            ax4.yaxis.set_major_locator(plt.MaxNLocator(4))
#            ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
#            divider = make_axes_locatable(ax4)
#            cax = divider.append_axes('bottom', size="5%", pad=0.05)
#            cb = plt.colorbar(im4, cax=cax, orientation="horizontal")
#            cb.set_label('Volumetric SM (%)')

            ### ----------------------------------------------------------------

#            # plot CONAE_HS interpolado mapa

#            fig4, ax4 = plt.subplots()
#            im4= ax4.imshow(band_matchCONAE_HS, extent=[xmin,xmax,ymin,ymax], cmap=my_cmap1, clim=(5, 45))

            plt.show()

            #im1 = ax.imshow(filtWater, cmap=my_cmap)
            #maskNDVI2 = ma.masked_where(filtWater == 0,filtWater)
            #im1 = ax.imshow(maskNDVI2, cmap=my_cmap)
            #im1 = ax.imshow(band_matchCity, cmap=my_cmap2)

            #mapSM_MLP = mapSM_MLP*maskNDVI

            functions.createHDFfile(path, nameFileMLR, 'ENVI', mapSM_MLR, c, r, GeoTSar, ProjectSar)
            functions.createHDFfile(path, nameFileMARS, 'ENVI', mapSM_MARS, c, r, GeoTSar, ProjectSar)
            functions.createHDFfile(path, nameFileMLP, 'ENVI', mapSM_MLP, c, r, GeoTSar, ProjectSar)


    print("FIN")


