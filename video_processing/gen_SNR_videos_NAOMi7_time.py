# %%
import sys
import os
import random
import time
import glob
import numpy as np
import math
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, r'C:\Users\Yijun\Documents\GitHub\Shallow-UNet-Neuron-Segmentation_SUNS') # the path containing "suns" folder
# os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video
# from suns.PreProcessing.generate_masks import generate_masks_from_traces
# from suns.train_CNN_params import train_CNN, parameter_optimization_cross_validation


# %%
if __name__ == '__main__':
    # folder of the raw videos
    folder = sys.argv[1]
    rate_hz = int(folder.split('_')[1][:-2]) # 30 # frame rate of the video
    dir_video = 'F:\\NAOMi\\{}\\'.format(folder)
    # rate_hz = 10 # int(sys.argv[1]) # frame rate of the video
    # dir_video = 'F:\\NAOMi\\100s_{}Hz_N=400_40mW_noise10+23\\'.format(rate_hz) 
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['Video_'+str(x) for x in list(range(0,10))]
    sub_folder = '' # e.g. 'noSF'
    # %% setting parameters
    # Dimens = (256,256) # lateral dimensions of the video
    # nn = 3000 # number of frames used for preprocessing. 
    #     # Can be slightly larger than the number of frames of a video
    Mag = 0.785 # 1 # spatial magnification compared to ABO videos.
    Table_time = np.zeros((len(list_Exp_ID)))

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.

    # %% set folders
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 
    dir_parent = dir_video + sub_folder + '\\' # folder to save all the processed data
    dir_network_input = dir_parent + 'SNR video\\' # folder of the SNR videos
    # dir_mask = dir_parent + 'temporal_masks({})\\'.format(thred_std) # foldr to save the temporal masks

    # if not os.path.exists(dir_network_input):
    #     os.makedirs(dir_network_input) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    # (rows, cols) = Dimens # size of the original video
    # rowspad = math.ceil(rows/8)*8  # size of the network input and output
    # colspad = math.ceil(cols/8)*8

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation
    # list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.

    # filename_TF_template = 'NAOMi_tempolate.h5'
    # h5f = h5py.File(filename_TF_template,'r')
    # Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    # Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # Poisson_filt = np.array([1], dtype = 'single')
    # h5f = loadmat('filter_template 30Hz jGCaMP7s.mat')
    # fs_template = 30
    h5f = loadmat('filter_template 100Hz {}_ind_con=10.mat'.format(folder.split('_')[-1]))
    fs_template = 100
    Poisson_template = h5f['template'].squeeze()
    peak = Poisson_template.argmax()
    length = Poisson_template.shape
    xp = np.arange(-peak,length-peak,1)/fs_template
    x = np.arange(np.round(-peak*rate_hz/fs_template), np.round((length-peak)*rate_hz/fs_template)+1, 1)/rate_hz
    Poisson_filt = np.interp(x,xp,Poisson_template)
    Poisson_filt = Poisson_filt[Poisson_filt>=(Poisson_filt.max()*np.exp(-1))].astype('float32')
    Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
    # Alternative temporal filter kernel using a single exponential decay function
    # rise = 0.018 # rising time constant (unit: second)
    # decay = 0.2 # decay time constant (unit: second)
    # leng_tf = np.ceil(rate_hz*decay)+1
    # decay_part = np.exp(-np.arange(leng_tf)/rate_hz/decay)
    # rise_part = 1-np.exp(-np.arange(leng_tf)/rate_hz/rise)
    # Poisson_filt = rise_part * decay_part
    # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
    
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'Poisson_filt': Poisson_filt} # 'nn':nn, 
    # print(Poisson_filt)

    # pre-processing for training
    for (eid,Exp_ID) in enumerate(list_Exp_ID): # [:1]
        # h5_img = h5py.File(dir_video+Exp_ID+'.h5', 'r')
        # print(np.array(h5_img['mov']).min())
        network_input, start = preprocess_video(dir_video, Exp_ID, Params, None, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) # video_input, _ = 
        finish = time.time()
        Table_time[eid] = finish-start
        h5_video = os.path.join(dir_video, Exp_ID + '.h5')
        h5_file = h5py.File(h5_video,'r')
        (nframes, rows, cols) = h5_file['mov'].shape
        network_input = network_input[:,:rows,:cols]
        if dir_network_input:
            if not os.path.exists(dir_network_input):
                os.makedirs(dir_network_input) 
            f = h5py.File(os.path.join(dir_network_input, Exp_ID+".h5"), "w")
            f.create_dataset("network_input", data = network_input)
            f.close()
        # h5_img = h5py.File(dir_network_input+Exp_ID+'.h5', 'r')
        # video_input = np.array(h5_img['network_input'])

        # # %% Determine active neurons in all frames using FISSA
        # file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        # generate_masks_from_traces(file_mask, list_thred_ratio, dir_parent, Exp_ID)
        # # del video_input

    savemat(os.path.join(dir_network_input, "Table_time.mat"), {"Table_time": Table_time})
