from datetime import datetime
import numpy  as np
from matplotlib import pyplot as plt


def cloud_mask(band,cloud_band):


    out_band = np.where(np.bitwise_and(cloud_band,8)==8, np.nan, band)
    out_band = np.where(np.bitwise_and(cloud_band, 16) == 16, np.nan, out_band)

    return out_band




# 计算干边与湿边

def get_min_max(ndvi,lst):
    MiniList = []
    MaxList = []
    # 创建ndvi向量（0到1，间距为0.01）
    ndvi_vector = np.round(np.arange(0.01, 1.01, 0.01), 2)
    # 首先找到相同ndvi的lst值
    for val in ndvi_vector:
        lst_lst_val = []
        row, col = np.where((ndvi >= val-0.001) & (ndvi <= val+0.001))
        # 根据这些ndvi的位置，我们取温度值对应这些位置（行和列）
        for i,j in zip(row,col):
            if np.isfinite(lst[i, j]):
                lst_lst_val += [lst[i, j]]
            # 如果所需的ndvi有lst值，则计算最大值和最小值
        if lst_lst_val != []:
            lst_min_val = np.min(lst_lst_val)
            lst_max_val = np.max(lst_lst_val)
        else:
            lst_min_val = np.nan
            lst_max_val = np.nan

        # 找到的值被添加到MiniList和MaxList列表中
        MiniList += [lst_min_val]
        MaxList  += [lst_max_val]
    return MiniList,MaxList

def fit(MiniList,MaxList):
    ndvi_vector = np.round(np.arange(0.01, 1.01, 0.01), 2)
    MiniList_fin = []
    ndvi_fin = []
    for i, val in enumerate(MiniList):
        if np.isfinite(val):
            MiniList_fin += [val]
            ndvi_fin += [ndvi_vector[i]]
    dryPfit = np.polyfit(ndvi_fin[14:89], MiniList_fin[14:89], 1)
    MaxList_fin = []
    ndvi_fin = []
    for i, val in enumerate(MaxList):
        if np.isfinite(val):
            MaxList_fin += [val]
            ndvi_fin += [ndvi_vector[i]]
    wetPfit = np.polyfit(ndvi_fin[14:89], MaxList_fin[14:89], 1)
    return dryPfit,wetPfit


def compute_sm(ndvi, srt, MinPfit, MaxPfit):
    a1, b1 = MaxPfit
    a2, b2 = MinPfit

    srtw = b1 + (a1 * ndvi)
    srtd = b2 + (a2 * ndvi)

    w = (srt - srtd) / (srtw - srtd)
    w=np.where((w<0)|(w>1),np.nan,w)

    return w




def plot_scatter(ndvi, lst, MiniList, MaxList, MinPfit, MaxPfit, scatter_file=None):
    ndvi_vector = np.round(np.arange(0.01, 1.01, 0.01), 2)
    a1, b1 = MaxPfit
    a2, b2 = MinPfit
    linhamax = [b1 + (a1 * 0), b1 + (a1 * 1)]
    linhamin = [b2 + (a2 * 0), b2 + (a2 * 1)]

    plt.plot(ndvi.ravel(), lst.ravel(), "+", color='dimgray', markersize=4)
    plt.plot(ndvi_vector[14:89], MiniList[14:89], '+', color='b')
    plt.plot(ndvi_vector[14:89], MaxList[14:89], '+', color='r')
    if a1 > 0:
        plt.plot([0, 1], linhamax, color='r', markersize=8, \
                 label=f"Tsmax = {'%.1f' % b1} + {'%.1f' % abs(a1)} * ndvi")
    else:

        plt.plot([0, 1], linhamax, color='r', markersize=8, \
                 label=f"Tsmax = {'%.1f' % b1} - {'%.1f' % abs(a1)} * ndvi")
    if a2 > 0:
        plt.plot([0, 1], linhamin, color='b', markersize=8, \
                 label=f"Tsmin = {'%.1f' % b2} + {'%.1f' % abs(a2)} * ndvi")
    else:
        plt.plot([0, 1], linhamin, color='b', markersize=8, \
                 label=f"Tsmin = {'%.1f' % b2} - {'%.1f' % abs(a2)} * ndvi")
    plt.legend(loc='upper right', fontsize=12)
    plt.ylim(0,30)
    plt.xlabel("ndvi")
    plt.ylabel("srt")
    plt.title("ndvi vs srt Scatterplot")
    if scatter_file is not None:
        plt.savefig(scatter_file)
    plt.show()

