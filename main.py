import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from osgeo import gdal
from base_funtion import *
import os

path=r'E:\影像数据\landsat_sr'
out_name=r'E:\定量遥感产品\土壤含水量\sm.tif'


qa_band = glob(os.path.join(path,'*QA_PIXEL.TIF'))[0]
ref3_path=gdal.Open(glob(os.path.join(path,'*SR_B3.TIF'))[0])
ref4_path=gdal.Open(glob(os.path.join(path,'*SR_B4.TIF'))[0])
ref5_path=gdal.Open(glob(os.path.join(path,'*SR_B5.TIF'))[0])
swir_path=gdal.Open(glob(os.path.join(path,'*SR_B7.TIF'))[0])

inds = gdal.Open(glob(os.path.join(path,'*SR_B5.TIF'))[0])
in_band = inds.GetRasterBand(1)
x_size = in_band.XSize
y_size = in_band.YSize

qa_array = gdal.Open(qa_band).ReadAsArray().astype(int)
ref3_array=ref3_path.GetRasterBand(1).ReadAsArray().astype(float)*float(2.0000E-05)-0.1
ref4_array=ref4_path.GetRasterBand(1).ReadAsArray().astype(float)*float(2.0000E-05)-0.1
ref5_array=ref5_path.GetRasterBand(1).ReadAsArray().astype(float)*float(2.0000E-05)-0.1
ndvi=(ref5_array-ref4_array)/(ref5_array+ref4_array)
ndwi=(ref3_array - ref5_array) / (ref3_array + ref5_array)

ndvi=np.where((ndvi<0)|(ndvi>1),np.nan,ndvi)


swir_array=swir_path.GetRasterBand(1).ReadAsArray().astype(float)*float(2.0000E-05)-0.1
swir_array=cloud_mask(swir_array,qa_array)

srt=((1-swir_array)**2)/(2*swir_array)
srt=np.where(srt>20,np.nan,srt)

MiniList,MaxList=get_min_max(ndvi,srt)
dryPfit,wetPfit=fit(MiniList,MaxList)

sm=compute_sm(ndvi, srt, dryPfit, wetPfit)
sm=cloud_mask(sm,qa_array)
sm=np.where(ndwi > 0.05, 1, sm)

driver = inds.GetDriver()
sm_outimg = driver.Create(out_name, x_size, y_size, 1, gdal.GDT_Float32)
sm_outimg.SetGeoTransform(inds.GetGeoTransform())
sm_outimg.SetProjection(inds.GetProjection())
sm_outBand = sm_outimg.GetRasterBand(1)
sm_outBand.WriteArray(sm)
sm_outBand.FlushCache
sm_outimg = None