# %%
import sys
import os
import math
import numpy as np
import time
import h5py

from scipy.io import savemat, loadmat
# import multiprocessing as mp
# import fissa
from r_neuropil import estimate_contamination_ratios
from roi_masks import calculate_roi_and_neuropil_traces, create_roi_mask
from demixer import demix_time_dep_masks # _small
# if (sys.version_info.major+sys.version_info.minor/10)>=3.8
# try:
#     from multiprocessing.shared_memory import SharedMemory
#     # from traces_from_masks_mp_share import bgtraces_from_masks, traces_from_masks, traces_bgtraces_from_masks
#     # from traces_from_masks_mp_share import traces_bgtraces_from_masks
#     from traces_from_masks_mp_shm_nobg import traces_from_masks
#     # from traces_from_masks_mp_shm_nobg_com import traces_from_masks
#     # from traces_from_masks_nooverlap_mp_shm import traces_bgtraces_from_masks
#     # from use_nmfunmix_mp_diag_v1_shm_MSE import use_nmfunmix
# except:
#     raise ImportError('No SharedMemory module. Please use Python version >=3.8, or use memory mapping instead of SharedMemory')

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')


# %%
if __name__ == '__main__':
    list_Exp_ID = ['Video_'+str(x) for x in list(range(0,10))]
    Table_time = np.zeros((len(list_Exp_ID),3))
    video_type = sys.argv[2] # 'SNR' # 'Raw' # 
    dir_video = 'F:\\NAOMi\\'+sys.argv[1]+'\\' # '100s_30Hz_100+10'

    if video_type == 'SNR':
        varname = 'network_input' # 
        dir_video_SNR = os.path.join(dir_video, 'SNR video')
    else:
        varname = 'mov' # 
        dir_video_SNR = dir_video
    dir_masks = os.path.join(dir_video, 'GT Masks')
    dir_traces = os.path.join(dir_video, 'traces_AllenSDK_'+video_type)
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
    border = [0,0,0,0]

    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        start = time.time()
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        file_video = h5py.File(filename_video, 'r')
        (T, Lx, Ly) = video_shape = file_video[varname].shape
        video_dtype = file_video[varname].dtype
        video = file_video[varname]
        np.array(video).max()
        # nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
        # shm_video = SharedMemory(create=True, size=nbytes_video)
        # video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
        # video[:] = file_video[varname][()].ravel()
        # video = video.reshape(file_video[varname].shape)
        # # video_name = shm_video.name
        # file_video.close()

        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        (ncells, Lxm, Lym) = masks_shape = Masks.shape
        roi_mask_list = [create_roi_mask(Lym, Lxm, border, roi_mask=mask, label=str(m)) \
            for (m,mask) in enumerate(Masks)]
        # shm_masks = SharedMemory(create=True, size=Masks.nbytes)
        # FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
        # FinalMasks[:] = Masks.ravel()
        # FinalMasks = FinalMasks.reshape(Masks.shape)
        # masks_sum = FinalMasks.astype('uint8').sum(0)
        # masks_name = shm_masks.name
        # del Masks
        finish = time.time()
        print(finish - start)

        # Calculate traces and background traces
        start = time.time()
        (roi_traces, neuropil_traces, exclusions) = calculate_roi_and_neuropil_traces(video, roi_mask_list, border)
        # raw_traces = traces_from_masks(shm_video, video_dtype, video_shape, shm_masks, masks_shape) # , masks_sum
        # raw_traces, comx, comy = traces_from_masks(shm_video, video_dtype, video_shape, shm_masks, masks_shape) # , masks_sum
        finish_trace = time.time()
        print('trace calculation time: {} s'.format(finish_trace - start))

        # Use AllenSDK to calculate the decontaminated traces of neural activities. 
        # start = time.time()
        (unmixed_traces, drop_frames) = demix_time_dep_masks(roi_traces, video, Masks, -1)
        # (unmixed_traces, drop_frames) = demix_time_dep_masks_small(raw_traces.T, video, FinalMasks, comx, comy, -1)
        finish_demix = time.time()
        print('Demixing time: {} s'.format(finish_demix-finish_trace))

        # Calculate the neuropil subtraction ratio
        # r_dict = estimate_contamination_ratios(unmixed_traces, neuropil_traces)
        # r = r_dict['r']
        # compensated_traces = unmixed_traces - r* neuropil_traces
        compensated_traces = np.zeros_like(unmixed_traces)
        r = np.zeros(ncells)
        for nn in range(ncells):
            r_dict = estimate_contamination_ratios(unmixed_traces[nn], neuropil_traces[nn])
            r[nn] = r_dict['r']
            compensated_traces[nn] = unmixed_traces[nn] - r[nn] * neuropil_traces[nn]
        finish = time.time()
        print('neuropil subtraction time: {} s'.format(finish - finish_demix))
        
        Table_time[ind_Exp,0] = finish_trace-start
        Table_time[ind_Exp,1] = finish_demix-finish_trace
        Table_time[ind_Exp,2] = finish-finish_demix
        savemat(os.path.join(dir_traces, Exp_ID+".mat"), {"compensated_traces": compensated_traces, \
            "r":r, "roi_traces":roi_traces, "unmixed_traces": unmixed_traces, "neuropil_traces": neuropil_traces})

        savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time})

