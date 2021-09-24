# %%
import sys
import os
import numpy as np
import time
import h5py
from scipy.io import savemat, loadmat

sys.path.insert(0, 'C:\\Other methods\\AllenSDK-master') # The folder containing the Allen SDK code
# from r_neuropil import estimate_contamination_ratios
# from roi_masks import calculate_roi_and_neuropil_traces, create_roi_mask
# from demixer import demix_time_dep_masks
from allensdk.brain_observatory.r_neuropil import estimate_contamination_ratios
from allensdk.brain_observatory.roi_masks import calculate_roi_and_neuropil_traces, create_roi_mask
from allensdk.brain_observatory.demixer import demix_time_dep_masks

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
np.seterr(divide='ignore',invalid='ignore')


# %%
if __name__ == '__main__':
    list_Exp_ID = ['501484643','501574836','501729039','502608215','503109347',
        '510214538','524691284','527048992','531006860','539670003']
    Table_time = np.zeros((len(list_Exp_ID),3))
    video_type = sys.argv[1] # 'SNR' # 'Raw' # 

    # dir_video = 'D:\\ABO\\20 percent 200' 
    dir_video = '..\\data\\ABO\\'
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

        finish = time.time()
        print(finish - start)

        # Calculate traces and background traces
        start = time.time()
        (roi_traces, neuropil_traces, exclusions) = calculate_roi_and_neuropil_traces(video, roi_mask_list, border)
        finish_trace = time.time()
        print('trace calculation time: {} s'.format(finish_trace - start))

        # Use AllenSDK to calculate the decontaminated traces of neural activities. 
        (unmixed_traces, drop_frames) = demix_time_dep_masks(roi_traces, video, Masks, -1)
        finish_demix = time.time()
        print('Demixing time: {} s'.format(finish_demix-finish_trace))

        # Calculate the neuropil subtraction ratio
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

