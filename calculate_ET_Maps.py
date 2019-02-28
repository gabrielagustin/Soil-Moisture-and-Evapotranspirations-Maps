# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Mon Sep 10 14:37:54 2018

@author: gag
"""


import numpy.ma as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error



import functions



def calculateETMaps():

    dir = "..."
    path = "/.../"+dir+"/.../"

    fechas= []
    fechas.append("2016_05_15")
    fechas.append("2016_05_07")
    fechas.append("2016_06_16")
    fechas.append("2016_07_10")

    pathOut = "/.../"+dir+"/.../CreatedMaps/"
    Ta = []
    HR = []
    PP = []
    sigma0 = []

    for i in range(0,len(fechas)):
    #for i in range(0,1):
        print(fechas[i])


        ### ET modis
        fileETModis = "/media/"+dir+"/TOURO Mobile/ET/"+fechas[i]+"/MYD16A2/MYD16A2_reprojected.data/ET_500m.img"        
        src_ds_ETModis, bandETModis, GeoTETModis, ProjectETModis = functions.openFileHDF(fileETModis, 1)
        
        ### SMAP resolucion 36km
        fileSM = "/media/"+dir+"/TOURO Mobile/ET/SM_36km/"+fechas[i]+"/SM.dat" 
        src_ds_SM, bandSM, GeoTSM, ProjectSM = functions.openFileHDF(fileSM, 1)
        print("tamanio SM SMAP:" + str(bandSM.shape))

        ### observed variables
        fileRn = "/media/"+dir+"/TOURO Mobile/ET/"+fechas[i]+"/RN/mapa_RN.asc"
        src_ds_Rn, bandRn, GeoTRn, ProjectRn = functions.openFileHDF(fileRn, 1)
        

        fileG = "/media/"+dir+"/TOURO Mobile/ET/"+fechas[i]+"/G/mapa_G.asc"
        src_ds_G, bandG, GeoTG, ProjectG = functions.openFileHDF(fileG, 1)        
        
        fileDelta = "/media/"+dir+"/TOURO Mobile/ET/"+fechas[i]+"/Delta/mapa_delta.asc"
        src_ds_Delta, bandDelta, GeoTDelta, ProjectDelta = functions.openFileHDF(fileDelta, 1)

        #### real ET observed
        fileETObs = "/media/"+dir+"/TOURO Mobile/ET/"+fechas[i]+"/ETobs/mapa_ETobs.asc"
        src_ds_ETObs, bandETObs, GeoTETObs, ProjectETObs = functions.openFileHDF(fileDelta, 1)

        nameFileET = "mapa_ET_"+str(fechas[i])

        ### se cambian las resoluciones de todas las imagenes a la de la sar
        type = "Nearest"
#        type = "Bilinear"
        nRow, nCol = bandSM.shape


#        fig, ax = plt.subplots()
#        ax.imshow(bandSM, interpolation='None',cmap=cm.gray)


        data_src = src_ds_ETObs
        data_match = src_ds_SM
        match = functions.matchData(data_src, data_match, type, nRow, nCol)
        band_matchETObs = match.ReadAsArray()
#        fig, ax = plt.subplots()
#        ax.imshow(bandETObs*1000, interpolation='None',cmap=cm.gray)

       data_src = src_ds_ETModis
       data_match = src_ds_SM
       match = functions.matchData(data_src, data_match, type, nRow, nCol)
       band_matchET = match.ReadAsArray()
#        fig, ax = plt.subplots()
#        ax.imshow(band_matchET, interpolation='None',cmap=cm.gray)


        data_src = src_ds_Rn
        data_match = src_ds_SM
        match = functions.matchData(data_src, data_match, type, nRow, nCol)
        band_matchRn = match.ReadAsArray()
#        fig, ax = plt.subplots()
#        ax.imshow(band_matchRn, interpolation='None',cmap=cm.gray)

        data_src = src_ds_G
        data_match = src_ds_SM
        match = functions.matchData(data_src, data_match, type, nRow, nCol)
        band_matchG = match.ReadAsArray()
#        fig, ax = plt.subplots()
#        ax.imshow(band_matchG, interpolation='None',cmap=cm.gray)

        data_src = src_ds_Delta
        data_match = src_ds_SM
        match = functions.matchData(data_src, data_match, type, nRow, nCol)
        band_matchDelta = match.ReadAsArray()
#        fig, ax = plt.subplots()
#        ax.imshow(band_matchDelta, interpolation='None',cmap=cm.gray)


################################################################################
######## here goes the equation to calculate the ET       
################################################################################

#        fig, ax = plt.subplots()
#        ax.imshow(mapET, interpolation='None',cmap=cm.gray)
#        plt.show()


        ##my_cmap = cm.Blues
        ##my_cmap.set_under('k', alpha=0)
        ##my_cmap1 = cm.Greens
        ##my_cmap1.set_under('k', alpha=0)
        ##my_cmap2 = cm.OrRd
        ##my_cmap2.set_under('k', alpha=0)
        #my_cmap3 = cm.Oranges
        my_cmap3 = cm.terrain
        my_cmap3.set_under('k', alpha=0)

        transform = GeoTSM
        xmin,xmax,ymin,ymax=transform[0],transform[0]+transform[1]*src_ds_SM.RasterXSize,transform[3]+transform[5]*src_ds_SM.RasterYSize,transform[3]
        #print xmin
        #print xmax

        path = "/.../ET_modelado_36km/"
        nameFileET = "ET_modelado_"+str(fechas[i])
        ### mapas ET modelados
        fig, ax = plt.subplots()
        mapET = (mapET-np.min(mapET)) /(np.max(mapET)-np.min(mapET))
        ### guarda mapa        
        functions.createHDFfile(path, nameFileET, 'ENVI', mapET, nCol, nRow, GeoTSM, ProjectSM)        

        
        im1 = ax.imshow(mapET, interpolation='none', cmap=plt.get_cmap('gray'), extent=[xmin,xmax,ymin,ymax], clim=(0, 1))
        ax.xaxis.tick_top()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size="5%", pad=0.05)
        cb = plt.colorbar(im1, cax=cax, orientation="horizontal")
        cb.set_label('Evapotranspiration (W/m^2)')
        print ("ET modelado:")
        print ("Max:" + str(np.max(mapET)))
        print ("Min:" + str(np.min(mapET)))
        print ("Std:" + str(np.std(mapET)))                
               
        ### mapas ET observado interpolado
        fig, ax = plt.subplots()
        band_matchETObs = band_matchETObs*1000
        band_matchETObs = (band_matchETObs- np.min(band_matchETObs)) /(np.max(band_matchETObs)-np.min(band_matchETObs))
        im0 = ax.imshow(band_matchETObs, cmap=plt.get_cmap('gray'), extent=[xmin,xmax,ymin,ymax], interpolation='none', clim=(0, 1))
        ax.xaxis.tick_top()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size="5%", pad=0.05)
        cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
        cb.set_label('Evapotranspiration (W/m^2)')
        #cb.set_clim(vmin=5, vmax=50)

        print ("ET observado:")
        print ("Max:" + str(np.max(band_matchETObs)))
        print ("Min:" + str(np.min(band_matchETObs)))
        print ("Std:" + str(np.std(band_matchETObs)))

        print("Error entre ET modelado y observado")

        mse_noise= mean_squared_error(y_true = band_matchETObs , y_pred = mapET)
        mse_noise = np.sqrt(mse_noise)
        #mse_noise= compare_mse(bandET_modis, bandET_modeled)        
        ssim_noise = ssim(band_matchETObs.flatten(), mapET.flatten())
        
        print("SSIM:" +str(ssim_noise))
        print("RMSE:" + str(mse_noise))


        fig, ax = plt.subplots()
        errorModelado = np.sqrt((mapET - band_matchETObs)**2)
        im0 = ax.imshow(errorModelado, cmap=plt.cm.jet, extent=[xmin,xmax,ymin,ymax], interpolation='none', clim=(0, 1))
        ax.xaxis.tick_top()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size="5%", pad=0.05)
        cb = plt.colorbar(im0, cax=cax, orientation="horizontal")
        cb.set_label('Error')
        #cb.set_clim(vmin=5, vmax=50)
        plt.show()

        


if __name__ == '__main__':
    calculateETMaps()
