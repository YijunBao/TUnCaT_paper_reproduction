import os
import sys
import numpy as np

from run_TUnCaT import run_TUnCaT


if __name__ == '__main__':
    # The folder containing the videos
    dir_video = '.'
    # A list of the name of the videos
    list_Exp_ID = ['c28_163_244']
    # A list of tested alpha.
    list_alpha = [1]
    # Method if there are multiple elements in "list_alpha". Can be 'multi' or 'one'
    alpha_option = 'multi' # 'one' # 
    # 'one' means the largest element providing non-trivial output traces will be used, 
    # which can be differnt for different neurons. It must be sorted in ascending order.
    # 'multi' means each element will be tested and saved independently.

    # The folder name (excluding the file name) containing the video
    dir_video_SNR = dir_video
    # The folder name (excluding the file name) containing the neuron masks
    dir_masks = dir_video
    # The folder to save the unmixed traces.
    dir_traces = os.path.join(dir_video, 'unmixed_traces')

    # Traces lower than this quantile are clipped to this quantile value.
    Qclip = 0  # 0.08 # 
    # Maximum pertentage of unmixed traces equaling to the trace minimum.
    th_pertmin = 1 # float(sys.argv[4])
    # The minimum value of the input traces after scaling and shifting. 
    epsilon = 0 # float(sys.argv[5])
    # Whether a direction requirement is applied.
    use_direction = False # bool(int(sys.argv[6]))
    # The temporal downsampling ratio.
    nbin = 1 # int(sys.argv[3]) # 
    # The method of temporal downsampling. can be 'downsample', 'sum', or 'mean'
    bin_option = 'downsample' # 'sum' # 'mean' # 

    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        print(Exp_ID)
        # The file path (including file name) of the video.
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        # The file path (including file name) of the neuron masks. 
        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        
        # run TUnCaT to calculate the unmixed traces of the marked neurons in the video
        run_TUnCaT(Exp_ID, filename_video, filename_masks, dir_traces, list_alpha, Qclip, \
            th_pertmin, epsilon, use_direction, nbin, bin_option, alpha_option)

