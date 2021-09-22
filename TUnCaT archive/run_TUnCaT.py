import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
import h5py
from utils import find_dataset

# if (sys.version_info.major+sys.version_info.minor/10)>=3.8
try:
    from multiprocessing.shared_memory import SharedMemory
    from traces_from_masks_numba_neighbors import traces_bgtraces_from_masks_numba_neighbors
    from traces_from_masks_mp_shm_neighbors import traces_bgtraces_from_masks_neighbors
    from use_nmfunmix_mp_diag_v1_shm_MSE_novideo import use_nmfunmix
except:
    raise ImportError('No SharedMemory module. Please use Python version >=3.8, or use memory mapping instead of SharedMemory')


def run_TUnCaT(Exp_ID, filename_video, filename_masks, dir_traces, list_alpha=[0], Qclip=0, \
        th_pertmin=1, epsilon=0, use_direction=False, nbin=1, bin_option='downsample'):
    ''' Unmix the traces of all neurons in a video, and obtain the unmixed traces and the mixing matrix. 
        The video is stored in "filename_video", and the neuron masks are stored in "filename_masks".
        The output traces will be stored in "dir_traces".
        If there are multiple elements in list_alpha, the largest element providing non-trivial output traces 
        will be used, which can be differnt for different neurons.
    Inputs: 
        Exp_ID (str): The name of the video.
        filename_video (str): The file path (including file name) of the video.
            The video file must be a ".h5" file. 
        filename_masks (str): The file path (including file name) of the neuron masks.
            The file must be a ".mat" file, and the masks are saved as variable "FinalMasks". 
        dir_traces (str): The folder to save the unmixed traces.
        list_alpha (list of float, default to [0]): A list of alpha to be tested.
            The elements should be sorted in ascending order.
        Qclip (float, default to 0): Traces lower than this quantile are clipped to this quantile value.
            Qclip = 0 means no clipping is applied. 
        th_pertmin (float, default to 1): Maximum pertentage of unmixed traces equaling to the trace minimum.
            th_pertmin = 1 means no requirement is applied. 
        epsilon (float, default to 0): The minimum value of the input traces after scaling and shifting. 
        use_direction (bool, default to False): Whether a direction requirement is applied.
            A direction requirement means the positive transients should be farther away from baseline than negative transients.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.

    Outputs:
        traces_nmfdemix (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace of a neuron.
        list_mixout (list of numpy.ndarray of float, shape = (n1,n1)): 
            Each element is the row-normalized mixing matrix for the NMF of each neuron.

    In addition to the returned variables, more outputs are stored under the folder "dir_traces".
        There are two sub-folders under this folder.
        The sub-folder "raw" stores the neuron traces and the background traces before NMF unmixing.
        The sub-folder "alpha={}" stores the same "traces_nmfdemix" and "list_mixout", 
        as well as three other quantities characterizing the NMF unmixing process:
            list_alpha_final (list of float): Each element is the final chosen alpha for the NMF of each neuron. 
                It might be one of the elements in "list_alpha", or a value smaller than the first element.
            list_MSE (list of numpy.ndarray of float, shape = (n1,)): 
                Each element is the mean squared error (NMF residual) between 
                the input traces and the NMF-reconstructed traces for the NMF of each neuron.
            list_n_iter (list of int): Each element is the number of iterations 
                to achieve NMF convergence for the NMF of each neuron.
    '''

    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
        
    start = time.time()
    # Create the shared memory object for the video
    file_video = h5py.File(filename_video, 'r')
    varname = find_dataset(file_video)
    (T, Lx, Ly) = video_shape = file_video[varname].shape
    video_dtype = file_video[varname].dtype
    nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
    shm_video = SharedMemory(create=True, size=nbytes_video)
    video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
    video[:] = file_video[varname][()].ravel()
    video = video.reshape(file_video[varname].shape)
    file_video.close()

    # Create the shared memory object for the masks
    try:
        file_masks = loadmat(filename_masks)
        Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
    except:
        file_masks = h5py.File(filename_masks, 'r')
        Masks = np.array(file_masks['FinalMasks']).astype('bool')
        file_masks.close()
    masks_shape = Masks.shape
    shm_masks = SharedMemory(create=True, size=Masks.nbytes)
    FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
    FinalMasks[:] = Masks.ravel()
    FinalMasks = FinalMasks.reshape(Masks.shape)
    del Masks
    finish = time.time()
    print('Data loading time: {} s'.format(finish - start))

    # Calculate traces, background, and outside traces
    start = time.time()
    if FinalMasks.sum()*T >= 7e7: # Using multiprocessing is faster for large videos
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_neighbors(shm_video, video_dtype, \
            video_shape, shm_masks, masks_shape, FinalMasks)
    else: # Using numba is faster for small videos
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
            video, FinalMasks)

    # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
    dir_trace_raw = os.path.join(dir_traces, "raw")
    finish = time.time()
    print('Trace calculation time: {} s'.format(finish - start))
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces})

    # Apply NMF to unmix the background-subtracted traces
    start = time.time()
    traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = use_nmfunmix(traces, bgtraces, outtraces, \
        list_neighbors, list_alpha, Qclip, th_pertmin, epsilon, use_direction, nbin, bin_option)
    finish = time.time()
    print('Unmixing time: {} s'.format(finish - start))

    # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
    if len(list_alpha) > 1:
        dir_trace_unmix = os.path.join(dir_traces, "list_alpha={}".format(str(list_alpha)))
    else:
        dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(list_alpha[0]))
    if not os.path.exists(dir_trace_unmix):
        os.makedirs(dir_trace_unmix)        
    savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
        "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

    # Unlink shared memory objects
    shm_video.close()
    shm_video.unlink()
    shm_masks.close()
    shm_masks.unlink()

    return traces_nmfdemix, list_mixout


def run_TUnCaT_multi_alpha(Exp_ID, filename_video, filename_masks, dir_traces, list_alpha=[0], Qclip=0, \
        th_pertmin=1, epsilon=0, use_direction=False, nbin=1, bin_option='downsample'):
    ''' Unmix the traces of all neurons in a video, and obtain the unmixed traces and the mixing matrix. 
        The video is stored in "filename_video", and the neuron masks are stored in "filename_masks".
        The output traces will be stored in "dir_traces".
        If there are multiple elements in list_alpha, each element will be tested and saved independently.
    Inputs: 
        Exp_ID (str): The name of the video.
        filename_video (str): The file path (including file name) of the video.
            The video file must be a ".h5" file. 
        filename_masks (str): The file path (including file name) of the neuron masks.
            The file must be a ".mat" file, and the masks are saved as variable "FinalMasks". 
        dir_traces (str): The folder to save the unmixed traces.
        list_alpha (list of float, default to [0]): A list of alpha to be tested.
        Qclip (float, default to 0): Traces lower than this quantile are clipped to this quantile value.
            Qclip = 0 means no clipping is applied. 
        th_pertmin (float, default to 1): Maximum pertentage of unmixed traces equaling to the trace minimum.
            th_pertmin = 1 means no requirement is applied. 
        epsilon (float, default to 0): The minimum value of the input traces after scaling and shifting. 
        use_direction (bool, default to False): Whether a direction requirement is applied.
            A direction requirement means the positive transients should be farther away from baseline than negative transients.
        nbin (int, default to 1): The temporal downsampling ratio.
            nbin = 1 means temporal downsampling is not used.
        bin_option (str, can be 'downsample' (default), 'sum', or 'mean'): 
            The method of temporal downsampling. 
            'downsample' means keep one frame and discard "nbin" - 1 frames for every "nbin" frames.
            'sum' means each binned frame is the sum of continuous "nbin" frames.
            'mean' means each binned frame is the mean of continuous "nbin" frames.

    Outputs:
        traces_nmfdemix (numpy.ndarray of float, shape = (T,n)): The resulting unmixed traces. 
            Each column is the unmixed trace of a neuron.
        list_mixout (list of numpy.ndarray of float, shape = (n1,n1)): 
            Each element is the row-normalized mixing matrix for the NMF of each neuron.

    In addition to the returned variables, more outputs are stored under the folder "dir_traces".
        There are two sub-folders under this folder.
        The sub-folder "raw" stores the neuron traces and the background traces before NMF unmixing.
        The sub-folder "alpha={}" stores the same "traces_nmfdemix" and "list_mixout", 
        as well as three other quantities characterizing the NMF unmixing process:
            list_alpha_final (list of float): Each element is the final chosen alpha for the NMF of each neuron. 
                It might be one of the elements in "list_alpha", or a value smaller than the first element.
            list_MSE (list of numpy.ndarray of float, shape = (n1,)): 
                Each element is the mean squared error (NMF residual) between 
                the input traces and the NMF-reconstructed traces for the NMF of each neuron.
            list_n_iter (list of int): Each element is the number of iterations 
                to achieve NMF convergence for the NMF of each neuron.
    '''

    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
        
    start = time.time()
    # Create the shared memory object for the video
    file_video = h5py.File(filename_video, 'r')
    varname = find_dataset(file_video)
    (T, Lx, Ly) = video_shape = file_video[varname].shape
    video_dtype = file_video[varname].dtype
    nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
    shm_video = SharedMemory(create=True, size=nbytes_video)
    video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
    video[:] = file_video[varname][()].ravel()
    video = video.reshape(file_video[varname].shape)
    file_video.close()

    # Create the shared memory object for the masks
    try:
        file_masks = loadmat(filename_masks)
        Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
    except:
        file_masks = h5py.File(filename_masks, 'r')
        Masks = np.array(file_masks['FinalMasks']).astype('bool')
        file_masks.close()
    masks_shape = Masks.shape
    shm_masks = SharedMemory(create=True, size=Masks.nbytes)
    FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
    FinalMasks[:] = Masks.ravel()
    FinalMasks = FinalMasks.reshape(Masks.shape)
    del Masks
    finish = time.time()
    print('Data loading time: {} s'.format(finish - start))

    # Calculate traces, background, and outside traces
    start = time.time()
    if FinalMasks.sum()*T >= 7e7: # Using multiprocessing is faster for large videos
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_neighbors(shm_video, video_dtype, \
            video_shape, shm_masks, masks_shape, FinalMasks)
    else: # Using numba is faster for small videos
        (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
            video, FinalMasks)

    # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
    dir_trace_raw = os.path.join(dir_traces, "raw")
    finish = time.time()
    print('Trace calculation time: {} s'.format(finish - start))
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces})

    for alpha in list_alpha:
        # Apply NMF to unmix the background-subtracted traces
        start = time.time()
        traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = use_nmfunmix(traces, bgtraces, outtraces, \
            list_neighbors, [alpha], Qclip, th_pertmin, epsilon, use_direction, nbin, bin_option)
        finish = time.time()
        print('NMF unmixing time: {} s'.format(finish - start))

        # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
        dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
        if not os.path.exists(dir_trace_unmix):
            os.makedirs(dir_trace_unmix)        
        savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
            "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

    # Unlink shared memory objects
    shm_video.close()
    shm_video.unlink()
    shm_masks.close()
    shm_masks.unlink()

    return traces_nmfdemix, list_mixout