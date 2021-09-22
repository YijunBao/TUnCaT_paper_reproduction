# %%
import sys
import os
import math
import numpy as np
import time
import h5py

from scipy.io import savemat, loadmat
import multiprocessing as mp
import fissa

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')


# %%
if __name__ == '__main__':
    list_alpha = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]  # 
    dir_video = 'E:\\OnePhoton videos\\cropped videos\\'
    list_Exp_ID = ['c25_59_228','c27_12_326','c28_83_210',
                'c25_163_267','c27_114_176','c28_161_149',
                'c25_123_348','c27_122_121','c28_163_244']
    Table_time = np.zeros((len(list_Exp_ID), len(list_alpha)+1))
    video_type = sys.argv[1]
    # video_type = 'SNR' # 'Raw' # 
    # eid_select = int(sys.argv[2])

    if video_type == 'SNR':
        dir_video_SNR = os.path.join(dir_video, 'SNR video')
    else:
        dir_video_SNR = dir_video
    dir_masks = os.path.join(dir_video, 'GT Masks merge')
    dir_traces = os.path.join(dir_video, 'traces_FISSA_'+video_type+'_merge_n_iter')
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
    dir_trace_raw = os.path.join(dir_traces, "raw")
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        

    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        # if ind_Exp > eid_select:
        #     continue
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        file_video = h5py.File(filename_video, 'r')
        video = np.array(file_video[list(file_video.keys())[0]])
        file_video.close()

        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            FinalMasks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            FinalMasks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()

        (nframesf, rowspad, colspad) = video.shape
        (ncells, rows, cols) = FinalMasks.shape
        # The lateral shape of "video_SNR" can be larger than that of "FinalMasks" due to padding in pre-processing
        # This step crop "video_SNR" to match the shape of "FinalMasks"
        video = video[:, :rows, :cols]

        # Use FISSA to calculate the decontaminated traces of neural activities. 
        folder_FISSA = os.path.join(dir_traces, 'FISSA')    
        start = time.time()
        experiment = fissa.Experiment([video], [FinalMasks.tolist()], folder_FISSA, ncores_preparation=1)
        experiment.separation_prep(redo=True)
        prep = time.time()
        print('Preparation time: {} s'.format(prep-start))
        Table_time[ind_Exp, -1] = prep-start

        for (ind_alpha, alpha) in enumerate(list_alpha):
            start = time.time()
            experiment = fissa.Experiment([video], [FinalMasks.tolist()], folder_FISSA, alpha=alpha, ncores_preparation=1)
            # experiment.separation_prep(redo=False)
            experiment.separate(redo_prep=False, redo_sep=True)
            finish = time.time()
            experiment.save_to_matlab()
            print('Separation time: {} s'.format(finish-start))
            Table_time[ind_Exp, ind_alpha] = finish-start

            # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
            if ind_alpha == 0:
                raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
                savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"raw_traces": raw_traces})

            # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
            unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
            list_n_iter = np.array([experiment.info[x][0]['iterations'] for x in range(ncells)])
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
            if not os.path.exists(dir_trace_unmix):
                os.makedirs(dir_trace_unmix)        
            savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"unmixed_traces": unmixed_traces, "list_n_iter":list_n_iter})

            # # Calculate median and median-based std to normalize each trace into SNR trace
            # # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
            # med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
            # med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
            # mu_unmix = med_unmix[0]
            # sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))


        savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time, 'list_alpha': list_alpha})

