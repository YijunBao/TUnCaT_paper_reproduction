#!/usr/bin/env python

# @package demos
#\brief      for the user/programmer to understand and try the code
#\details    all of other usefull functions (demos available on jupyter notebook) -*- coding: utf-8 -*-
#\version   1.0
#\pre       EXample.First initialize the system.
#\bug
#\warning
#\copyright GNU General Public License v2.0
#\date Created on Mon Nov 21 15:53:15 2016
#\author agiovann
# toclean

"""
Prepare ground truth built by matching with the results of CNMF
"""
# %%
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
# import multiprocessing as mp
import glob
import time
from scipy.io import loadmat, savemat

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import sys
sys.path.insert(0, 'C:\\Other methods\\CaImAn') # The folder containing the caiman code

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # "-1", "1"

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman import mmapping
from caiman.base.rois import mask_to_2d

import h5py
import cv2


if __name__ == "__main__":
    start = time.time()
    #%% GT 
    # dir_video = 'D:\\ABO\\20 percent 200\\'
    dir_video = '..\\..\\data\\ABO\\'
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    Table_time = np.zeros(len(list_Exp_ID))
    p = 1

    video_type = sys.argv[1] # 'Raw' # 'SNR' # 
    if video_type == 'SNR':
        varname = 'network_input' # 
        dir_video_SNR = os.path.join(dir_video, 'SNR video')
    else:
        varname = 'mov' # 
        dir_video_SNR = dir_video
    dir_masks = os.path.join(dir_video, 'GT Masks')
    dir_traces = os.path.join(dir_video, 'traces_CNMF_'+video_type+'_p'+str(p))
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 

    GTMask = {}
    for Exp_ID in list_Exp_ID:
        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        GTMask[Exp_ID] = Masks  

    Names_raw = glob.glob(dir_video_SNR+'\\*_memmap__*.mmap')


    for (cnt, Exp_ID) in enumerate(list_Exp_ID):
        print(Exp_ID)
        Name_mmap = [x for x in Names_raw if Exp_ID in x ]
        if len(Name_mmap) != 1:
            if len(Name_mmap) > 1:
                print('Multiple memory mapping files found. Delete dupllicate or incorrect files')
            filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
            fname_new = cm.save_memmap([filename_video],
                                    base_name =filename_video.rsplit('\\', 1)[-1][0:-3]+'_memmap_', 
                                    order='C')
        else:
            fname_new = Name_mmap[0]

        params_movie = {'fname': fname_new, 
                    'p': p,  # order of the autoregressive system
                    'merge_thresh': 1, # 0.8,  # merging threshold, max correlation allow
                    # 'final_frate': 30,
                    'gnb': 2,
                    # 'update_background_components': True,
                    'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                    #(to be used with one background per patch)
                    'swap_dim': False,  # for some movies needed
                    'kernel': None}

        roi_cons = GTMask[list_Exp_ID[cnt]]
        
        
        c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)
        
        # # % LOAD MEMMAP FILE
        Yr, dims, T = cm.load_memmap(fname_new)
        Yr.max()
        
        start = time.time()
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        Y = np.reshape(Yr, dims + (T,), order='F')
        m_images = cm.movie(images)
    #%

        radius = np.int(np.median(np.sqrt(np.sum(roi_cons, (1, 2)) / np.pi)))
        
        print(radius)
        print(roi_cons.shape)
        
        if params_movie['kernel'] is not None:  # kernel usually two
            kernel = np.ones(
                (radius // params_movie['kernel'], radius // params_movie['kernel']), np.uint8)
            roi_cons = np.vstack([cv2.dilate(rr, kernel, iterations=1)[
                                np.newaxis, :, :] > 0 for rr in roi_cons]) * 1.
            pl.imshow(roi_cons.sum(0), alpha=0.5)
        
        A_in = np.reshape(roi_cons.transpose(
            [2, 1, 0]), (-1, roi_cons.shape[0]), order='C')
        
        
        # % some parameter settings
        # order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
        p = params_movie['p']
        # merging threshold, max correlation allowed
        merge_thresh = params_movie['merge_thresh']
        
        # % Extract spatial and temporal components on patches
        # TODO: todocument
        if images.shape[0] > 10000:
            check_nan = False
        else:
            check_nan = True
        
        Cn = m_images.local_correlations(
            swap_dim=params_movie['swap_dim'], frames_per_chunk=1500)
        
        cnm = cnmf.CNMF(check_nan=check_nan, n_processes=n_processes, k=A_in.shape[-1], gSig=[radius, radius], \
            merge_thresh=params_movie['merge_thresh'], p=params_movie['p'], Ain=A_in.astype(np.bool),
            dview=dview, rf=None, stride=None, gnb=params_movie['gnb'], method_deconvolution='oasis', \
            border_pix=0, low_rank_background=params_movie['low_rank_background'], n_pixels_per_process=1000)
        cnm = cnm.fit(images)
        
        A = cnm.estimates.A
        C = cnm.estimates.C
        YrA = cnm.estimates.YrA
        b = cnm.estimates.b
        f = cnm.estimates.f
        snt = cnm.estimates.sn
        finish = time.time()
        print(('Number of components:' + str(A.shape[-1])))

        #% thredshold components
        min_size_neuro = 3 * 2 * np.pi
        max_size_neuro = (2 * radius)**2 * np.pi
        A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:, :].toarray(), dims, \
            medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True, se=None, ss=None, dview=dview)

        A_thr = A_thr > 0
        size_neurons = A_thr.sum(0).A.squeeze()
        idx_size_neuro = np.where((size_neurons > min_size_neuro)
                                & (size_neurons < max_size_neuro))[0]
        A_thr = A_thr[:, idx_size_neuro]

        roi_cons = GTMask[list_Exp_ID[cnt]]#name[0:3]]
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = cm.base.rois.nf_match_neurons_in_binary_masks(
            roi_cons, A_thr.toarray().reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1., thresh_cost=.5, min_dist=10,
            # roi_cons, A_thr[:, :].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1., thresh_cost=.7, min_dist=10,
            print_assignment=False, plot_results=False, Cn=Cn, labels=['GT', 'Offline'])
        # cnt +=1 
        #%
        savemat(os.path.join(dir_traces, Exp_ID + '.mat'), {'Cn':Cn,
            'tp_gt':tp_gt, 'tp_comp':tp_comp, 'fn_gt':fn_gt, 'fp_comp':fp_comp, 'performance_cons_off':performance_cons_off, 
            'idx_size_neuro_gt':idx_size_neuro, 'A_thr':A_thr,
            'A_gt':A, 'C_gt':C, 'b_gt':b, 'f_gt':f, 'YrA_gt':YrA, 'd1':d1, 'd2':d2, 'idx_components_gt':idx_size_neuro[tp_comp],
            'idx_components_bad_gt':idx_size_neuro[fp_comp], 'fname_new':fname_new}, do_compression=True)
        #%
        # np.savez(os.path.join(dirSave, os.path.split(fname_new)[1][:-4] + 'match_masks.npz'), Cn=Cn,
        #     tp_gt=tp_gt, tp_comp=tp_comp, fn_gt=fn_gt, fp_comp=fp_comp, performance_cons_off=performance_cons_off, idx_size_neuro_gt=idx_size_neuro, A_thr=A_thr,
        #     A_gt=A, C_gt=C, b_gt=b, f_gt=f, YrA_gt=YrA, d1=d1, d2=d2, idx_components_gt=idx_size_neuro[
        #         tp_comp],
        #     idx_components_bad_gt=idx_size_neuro[fp_comp], fname_new=fname_new)

        #% STOP CLUSTER and clean up log files
        cm.stop_server(dview=dview)
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        Table_time[cnt] = finish - start

    savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time})
