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
    folder = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f' # sys.argv[1]
    rate_hz = int(folder.split('_')[1][:-2]) # 30 # frame rate of the video
    # dir_video = 'F:\\NAOMi\\{}\\'.format(folder)
    dir_video = '..\\data\\NAOMi\\'
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['Video_'+str(x) for x in list(range(0,10))]
    sub_folder = ''
    # %% setting parameters
    Mag = 0.785 # spatial magnification compared to ABO videos.
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

    h5f = loadmat('../template/filter_template 100Hz {}_ind_con=10.mat'.format(folder.split('_')[-1]))
    fs_template = 100
    Poisson_template = h5f['template'].squeeze()
    peak = Poisson_template.argmax()
    length = Poisson_template.shape
    xp = np.arange(-peak,length-peak,1)/fs_template
    x = np.arange(np.round(-peak*rate_hz/fs_template), np.round((length-peak)*rate_hz/fs_template)+1, 1)/rate_hz
    Poisson_filt = np.interp(x,xp,Poisson_template)
    Poisson_filt = Poisson_filt[Poisson_filt>=(Poisson_filt.max()*np.exp(-1))].astype('float32')
    Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
    # dictionary of pre-processing parameters
    Params = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
        'Poisson_filt': Poisson_filt}

    # Generate SNR videos
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        network_input, start = preprocess_video(dir_video, Exp_ID, Params, None, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, prealloc=prealloc) # video_input, _ = 
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
