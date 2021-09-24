# %%
import os
import time
import numpy as np
import h5py
from scipy.io import savemat, loadmat

from preprocessing_functions import preprocess_video


# %%
if __name__ == '__main__':
    # folder of the raw videos
    # rate_hz = 20 # frame rate of the video
    # dir_video = 'E:\\OnePhoton videos\\cropped videos\\'
    dir_video = '..\\data\\1p\\'
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['c25_59_228','c27_12_326','c28_83_210',
                'c25_163_267','c27_114_176','c28_161_149',
                'c25_123_348','c27_122_121','c28_163_244']
    sub_folder = ''
    # %% setting parameters
    Mag = 0.5 # spatial magnification compared to ABO videos.
    Table_time = np.zeros((len(list_Exp_ID)))

    useSF=True # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.

    # %% set folders
    dir_GTMasks = dir_video + 'GT Masks\\FinalMasks_' 
    dir_parent = dir_video + sub_folder + '\\' # folder to save all the processed data
    dir_SNR = dir_parent + 'SNR video\\' # folder of the SNR videos

    if not os.path.exists(dir_SNR):
        os.makedirs(dir_SNR) 

    # %% set pre-processing parameters
    gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
    num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation

    # filename_TF_template = '../template/1P_spike_tempolate.h5'
    # h5f = h5py.File(filename_TF_template,'r')
    # Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
    filename_TF_template = '../template/1P_spike_tempolate.mat'
    h5f = loadmat(filename_TF_template)
    Poisson_filt = h5f['filter_tempolate'].squeeze().astype('float32')
    Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'Poisson_filt': Poisson_filt}

    # Generate SNR videos
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        network_input, start = preprocess_video(dir_video, Exp_ID, Params, None, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) # dir_network_input
        finish = time.time()
        Table_time[eid] = finish-start

        h5_video = os.path.join(dir_video, Exp_ID + '.h5')
        h5_file = h5py.File(h5_video,'r')
        (nframes, rows, cols) = h5_file['mov'].shape
        network_input = network_input[:,:rows,:cols]

        f = h5py.File(os.path.join(dir_SNR, Exp_ID+".h5"), "w")
        f.create_dataset("network_input", data = network_input)
        f.close()

    savemat(os.path.join(dir_SNR, "Table_time.mat"), {"Table_time": Table_time})
