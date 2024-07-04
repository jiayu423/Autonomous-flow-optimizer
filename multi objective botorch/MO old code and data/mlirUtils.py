from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import Optional, Union, Callable, Tuple, List
from sklearn.decomposition import PCA 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as linalg
import copy
from pathlib import Path

def loadConc(vol_added, stock_conc = np.array([10.49, 17.126, 17.48, 10.23]), isFirst=False): 

    # before dilution
    conc1 = np.zeros(vol_added.shape)
    for i in range(conc1.shape[0]): 
        conc1[i, :] = stock_conc * vol_added[i, :] / (5 + np.sum(vol_added[i, :]))
        
    # after dilution 
    if isFirst: 
        mod = 4
    else: 
        mod = 0
    conc2 = np.zeros((vol_added.shape[0]-mod, 4))
    for i in range(conc2.shape[0]): 
        conc2[i, :] = conc1[i, :] / 2
        
    conc_train = np.vstack((conc1, conc2))

    return conc_train

def loadSpec(folder_name): 

    entries = Path(folder_name)
    spectra_train = []
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[-1] == 'D':

            continue
        print(entry.name)
        spectra_train.append(pd.read_csv(folder_name + entry.name).to_numpy())
        
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is not 'd': 
            continue
        print(entry.name)
        spectra_train.append(pd.read_csv(folder_name + entry.name).to_numpy())

    spectra_train = np.array(spectra_train)

    return spectra_train


def loadDCMData(): 

    stock_conc = np.array([10.49, 17.126, 17.48, 10.23]) # ac2o, etoh, acoh, etoac

    vol_added = np.array([[130.,  73.,  73.,  90.],
       [110.,  70.,  71., 100.],
       [105.,  64.,  79., 140.],
       [120.,  80.,  75., 135.], 
       [ 95.,  50.,  96., 165.],
       [125.,  60.,  71., 135.],
       [ 80.,  66.,  89., 145.],
       [125.,  65.,  69., 125.],
       [110.,  60.,  80., 130.],
       [125.,  78.,  58., 105.],
       [105.,  72.,  63., 130.],
       [120.,  69.,  60., 125.],       
       [125.,  93.,  69., 100.],
       [ 90.,  59.,  76., 170.],
       [160., 104.,  46.,  80.],
       [135.,  68.,  75.,  95.],
       [140.,  96.,  55., 125.],
       [135.,  92.,  53.,  90.],
       [130.,  57.,  76., 125.],
       [ 65.,  26., 114., 180.]]) / 1000

    # before dilution
    conc1 = np.zeros(vol_added.shape)
    for i in range(conc1.shape[0]): 
        conc1[i, :] = stock_conc * vol_added[i, :] / (5 + np.sum(vol_added[i, :]))
        
    # after dilution 
    conc2 = np.zeros((vol_added.shape[0]-4, 4))
    for i in range(conc2.shape[0]): 
        conc2[i, :] = conc1[i, :] / 2
        
    conc_train1 = np.vstack((conc1, conc2))

    vol_added = np.array([[235.,  67.,  69., 110.],
           [305., 116.,  45.,  85.],
           [225.,  86.,  81., 145.],
           [295., 106.,  65., 110.],
           [255.,  78.,  82., 125.],
           [245.,  82.,  42.,  95.],
           [290., 101.,  66., 100.],
           [210.,  37.,  93., 125.],
           [290., 111.,  25.,  70.],
           [205.,  69.,  80., 165.],
           [285.,  95.,  40.,  70.],
           [220.,  57.,  96., 155.],
           [250.,  83.,  71., 100.],
           [280.,  88.,  46.,  75.],
           [275.,  94.,  66.,  75.],
           [155.,  31., 117., 165.],
           [275.,  75.,  66., 130.],
           [270.,  86.,  49., 110.],
           [210.,  50.,  74., 135.],
           [245.,  87.,  67., 120.]]) / 1000

    # before dilution
    conc1 = np.zeros(vol_added.shape)
    for i in range(conc1.shape[0]): 
        conc1[i, :] = stock_conc * vol_added[i, :] / (5 + np.sum(vol_added[i, :]))

        # after dilution 
    conc2 = np.zeros((vol_added.shape[0], 4))
    for i in range(conc2.shape[0]): 
        conc2[i, :] = conc1[i, :] / 2
        
    conc_train2 = np.vstack((conc1, conc2))

    vol_added = np.array([[ 55.,  68.,  60.,  55.],
           [ 95.,  78.,  35.,  55.],
           [135., 126.,   0.,   5.],
           [ 70.,  86.,  31.,  30.],
           [105., 100.,  21.,  15.],
           [ 20.,  65.,  59.,  65.],
           [105.,  88.,   9.,  40.],
           [ 40.,  66.,  45.,  55.],
           [ 80.,  80.,  17.,  15.],
           [ 75.,  86.,  49.,  50.],
           [110., 100.,  0.,  25.],
           [ 40.,  65.,  53., 105.],
           [  5.,  46.,  63., 115.],
           [  0.,  41.,  61., 145.],
           [ 15.,  54.,  60., 105.],
           [ 60.,  58.,  45., 100.],
           [ 45.,  65.,  52., 100.],
           [ 80.,  92.,  13.,  65.],
           [ 10.,  69.,  70.,  95.],
           [ 30.,  76.,  24.,  75.]]) / 1000

    # before dilution
    conc1 = np.zeros(vol_added.shape)
    for i in range(conc1.shape[0]): 
        conc1[i, :] = stock_conc * vol_added[i, :] / (5 + np.sum(vol_added[i, :]))

        # after dilution 
    conc2 = np.zeros((vol_added.shape[0], 4))
    for i in range(conc2.shape[0]): 
        conc2[i, :] = conc1[i, :] / 2
        
    conc_train3 = np.vstack((conc1, conc2))  

    folder_name = 'Training data/'

    entries = Path(folder_name)
    spectra_train1 = []
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is 'd': 
            continue
        print(entry.name)
        spectra_train1.append(pd.read_csv(folder_name + entry.name).to_numpy())
        
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is not 'd': 
            continue
        print(entry.name)
        spectra_train1.append(pd.read_csv(folder_name + entry.name).to_numpy())

    spectra_train1 = np.array(spectra_train1)

    folder_name = 'mlir005-10/'

    entries = Path(folder_name)
    spectra_train2 = []
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is 'd': 
            continue
        print(entry.name)
        spectra_train2.append(pd.read_csv(folder_name + entry.name).to_numpy())
        
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is not 'd': 
            continue
        print(entry.name)

        spectra_train2.append(pd.read_csv(folder_name + entry.name).to_numpy())
    spectra_train2 = np.array(spectra_train2)

    folder_name = 'mlir005-11/'

    entries = Path(folder_name)
    spectra_train3 = []
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is 'd': 
            continue
        print(entry.name)
        spectra_train3.append(pd.read_csv(folder_name + entry.name).to_numpy())
        
    for entry in sorted(entries.iterdir()): 
        
        if entry.name[0] == '.' or entry.name[-5] is not 'd': 
            continue
        print(entry.name)
        spectra_train3.append(pd.read_csv(folder_name + entry.name).to_numpy())
        
    spectra_train3 = np.array(spectra_train3)
    
    return conc_train1, conc_train2, conc_train3, spectra_train1, spectra_train2, spectra_train3


def CI(gp, X_test, n_samples): 
    y_hat_samples = gp.sample_y(X_test, n_samples=n_samples)
    
#     print(y_hat_samples.shape)
    y_hat_samples = y_hat_samples.reshape((X_test.shape[0], n_samples))
    
    y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
    y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()
    
    return y_hat, y_hat_sd


def atleast_2d(x):
    if len(x.shape)<2:
        return x.reshape(-1, 1)
    else:
        return x


def vtna_v2(reagent1, reagent2, product1, product2, order, leng, isPlot=False): 

    if not leng:
        leng = min(reagent1.shape[0], reagent2.shape[0])
    else: 
        pass
    
    leng = min(reagent1.shape[0], reagent2.shape[0])
    taxis1, taxis2 = np.zeros((leng-1, )), np.zeros((leng-1, ))

    for i in range(leng-1):  
        temp1, temp2 = np.zeros((leng-1, )), np.zeros((leng-1, ))
        temp1[i:] = ((reagent1[i+1]+reagent1[i])/2)**order 
        temp2[i:] = ((reagent2[i+1]+reagent2[i])/2)**order 
        taxis1 += temp1
        taxis2 += temp2

    if isPlot:
        plt.scatter(taxis1, product1[1:leng], facecolors='none',edgecolors='orange')
        plt.scatter(taxis2, product2[1:leng], marker='^', facecolors='none',edgecolors='b')
        plt.title(f'order={order}')
    
    return [taxis1, taxis2], [product1[1:leng], product2[1:leng]]


def score(order, args, leng):
    
    reagent1, reagent2, product1, product2 = args[0], args[1], args[2], args[3]
    
    abscissas, ordinates = vtna_v2(reagent1, reagent2, product1, product2, order, leng)
    
    data = pd.DataFrame()
    for abscissa, ordinate in zip(abscissas, ordinates):
        dataframe = pd.DataFrame(data={'x': abscissa, 'y': ordinate})
        data = data.append(dataframe)

    sorted_diff = data.sort_values('x').diff()
    error = sorted_diff.abs()['y'].sum()
    
    return float(error)


def minOrder(clist, od_range, leng=None):

    s = np.zeros(od_range.shape)
    for i, o in enumerate(od_range): 
        s[i] = score(o, clist, leng)
    od_min = od_range[np.where(s==s.min())[0][0]]
    
    return od_min


def drawRandomSampeles(gp, X_test, n_samples): 
    y_hat_samples = gp.sample_y(X_test, n_samples=n_samples)
    y_hat_samples = y_hat_samples.reshape((X_test.shape[0], n_samples))
    return y_hat_samples


def trainGP(xtrain, conc, l, normY=True): 

    # aniso = np.ones((n_comp, ))

    kernel = 1*RBF(l) + WhiteKernel()
    gp_ac2o = GaussianProcessRegressor(kernel=kernel, normalize_y=normY)
    gp_ac2o.fit(xtrain, conc[:, 0].reshape(-1, 1)) 

    kernel = 1*RBF(l) + WhiteKernel()
    gp_etoh = GaussianProcessRegressor(kernel=kernel, normalize_y=normY)
    gp_etoh.fit(xtrain, conc[:, 1].reshape(-1, 1)) 

    kernel = 1*RBF(l) + WhiteKernel()
    gp_acoh = GaussianProcessRegressor(kernel=kernel, normalize_y=normY)
    gp_acoh.fit(xtrain, conc[:, 2].reshape(-1, 1)) 

    kernel = 1*RBF(l) + WhiteKernel()
    gp_etoac = GaussianProcessRegressor(kernel=kernel, normalize_y=normY)
    gp_etoac.fit(xtrain, conc[:, 3].reshape(-1, 1)) 

    return gp_ac2o, gp_etoh, gp_acoh, gp_etoac


def loadRxnSpec(folder_name, spec_range): 
    
    entries = Path(folder_name)
    rxn = []
    for entry in sorted(entries.iterdir()): 

        if entry.name[0] == '.': 
            continue
        rxn.append(pd.read_csv(folder_name + entry.name).to_numpy())

    rxn = np.array(rxn)
    
#     rxn = rxn[:, ::2, :]

    rxn_data1 = rxn[:, spec_range[0]:spec_range[1], -1]
    rxn_data2= rxn[:, spec_range[2]:spec_range[3], -1]
    pca_data_reaction = np.hstack((rxn_data1, rxn_data2))
    
    return pca_data_reaction


def pred(folder_name, gp_list, pca_data_reaction, comp_list, n_samples):

    m1, sd1 = CI(gp_list[0], pca_data_reaction[:, :comp_list[0]], n_samples)
    m2, sd2 = CI(gp_list[1], pca_data_reaction[:, :comp_list[1]], n_samples)
    m3, sd3 = CI(gp_list[2], pca_data_reaction[:, :comp_list[2]], n_samples)
    m4, sd4 = CI(gp_list[3], pca_data_reaction[:, :comp_list[3]], n_samples)

    return [m1, sd1, m2, sd2, m3, sd3, m4, sd4]


def plotTraj(t, m, sd, nsd, c, l=None): 
    
    plt.scatter(t, m, c=c, label=l)
    plt.fill_between(x=t, y1=(m - nsd*sd), y2=(m + nsd*sd), color=c,  alpha=0.2)


def plotTimeCourses(conc_list, ind, nsd, isPlot=[True, False, False, True]): 

    cmap = ['r', 'g', 'b', 'm', 'y', 'violet', 'k']
    t_len = len(conc_list[ind][0])
    t = [i for i in range(t_len)]
    
    for i, plot in enumerate(isPlot): 

        if plot: 
            plotTraj(t, conc_list[ind][i*2], conc_list[ind][(i*2)+1], nsd, cmap[ind])