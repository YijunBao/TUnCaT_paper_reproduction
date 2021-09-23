import os
import sys
import time
import numpy as np
from scipy.io import savemat, loadmat
import h5py

# if (sys.version_info.major+sys.version_info.minor/10)>=3.8
from multiprocessing.shared_memory import SharedMemory
from traces_from_masks_numba_neighbors import traces_bgtraces_from_masks_numba_neighbors
from traces_from_masks_mp_shm_neighbors import traces_bgtraces_from_masks_shm_neighbors
from use_nmfunmix_mp_diag_v1_shm_MSE_novideo import use_nmfunmix


if __name__ == '__main__':
    # sys.argv = ['py', 'Raw', '1']
    list_alpha = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10] # 
    dir_video = 'E:\\OnePhoton videos\\cropped videos\\'
    list_Exp_ID = ['c25_59_228','c27_12_326','c28_83_210',
                'c25_163_267','c27_114_176','c28_161_149',
                'c25_123_348','c27_122_121','c28_163_244']
    Table_time = np.zeros((len(list_Exp_ID), len(list_alpha)+1))
    video_type = sys.argv[1] # 'Raw' # 'SNR' # 

    Qclip = 0
    nbin = int(sys.argv[2]) # 1 # 
    bin_option = 'downsample' # 'mean' # 'sum' # 
    if nbin == 1:
        addon = ''
    else:
        addon = '_'+bin_option +str(nbin)

    th_pertmin = 1
    epsilon = 0
    use_direction = False
    flexible_alpha = True
    addon += '_merge_novideounmix_r2_mixout'

    # Load video and FinalMasks
    if video_type == 'SNR':
        varname = 'network_input'
        dir_video_SNR = os.path.join(dir_video, 'SNR video')
    else:
        varname = 'mov'
        dir_video_SNR = dir_video
    dir_masks = os.path.join(dir_video, 'GT Masks merge')
    dir_traces = os.path.join(dir_video, 'traces_ours_'+video_type + addon)
    if not os.path.exists(dir_traces):
        os.makedirs(dir_traces) 
    dir_trace_raw = os.path.join(dir_traces, "raw")
    if not os.path.exists(dir_trace_raw):
        os.makedirs(dir_trace_raw)        
    
    for (ind_Exp, Exp_ID) in enumerate(list_Exp_ID):
        print(Exp_ID)
        start = time.time()
        filename_video = os.path.join(dir_video_SNR, Exp_ID + '.h5')
        file_video = h5py.File(filename_video, 'r')
        (T, Lx, Ly) = video_shape = file_video[varname].shape
        video_dtype = file_video[varname].dtype
        nbytes_video = int(video_dtype.itemsize * file_video[varname].size)
        shm_video = SharedMemory(create=True, size=nbytes_video)
        video = np.frombuffer(shm_video.buf, dtype = file_video[varname].dtype)
        video[:] = file_video[varname][()].ravel()
        video = video.reshape(file_video[varname].shape)
        file_video.close()

        filename_masks = os.path.join(dir_masks, 'FinalMasks_' + Exp_ID + '.mat')
        try:
            file_masks = loadmat(filename_masks)
            Masks = file_masks['FinalMasks'].transpose([2,1,0]).astype('bool')
        except:
            file_masks = h5py.File(filename_masks, 'r')
            Masks = np.array(file_masks['FinalMasks']).astype('bool')
            file_masks.close()
        (ncells, Lxm, Lym) = masks_shape = Masks.shape
        shm_masks = SharedMemory(create=True, size=Masks.nbytes)
        FinalMasks = np.frombuffer(shm_masks.buf, dtype = 'bool')
        FinalMasks[:] = Masks.ravel()
        FinalMasks = FinalMasks.reshape(Masks.shape)
        del Masks
        finish = time.time()
        print(finish - start)

        # Calculate traces and background traces
        start = time.time()
        if FinalMasks.sum()*T >= 7e7: # Use multiprocessing is faster for large videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_shm_neighbors(
                shm_video, video_dtype, video_shape, shm_masks, masks_shape, FinalMasks)
        else: # Use numba is faster for small videos
            (traces, bgtraces, outtraces, list_neighbors) = traces_bgtraces_from_masks_numba_neighbors(
                video, FinalMasks)

        finish = time.time()
        print('trace calculation time: {} s'.format(finish - start))
        Table_time[ind_Exp, -1] = finish-start

        # Save the raw traces into a ".mat" file under folder "dir_trace_raw".
        savemat(os.path.join(dir_trace_raw, Exp_ID+".mat"), {"traces": traces, "bgtraces": bgtraces})

        for (ind_alpha, alpha) in enumerate(list_alpha):
            print(Exp_ID, 'alpha =', alpha)
            # Do NMF unmixing
            start = time.time()
            traces_nmfdemix, list_mixout, list_MSE, list_final_alpha, list_n_iter = use_nmfunmix(
                traces, bgtraces, outtraces, list_neighbors, [alpha], Qclip, \
                th_pertmin, epsilon, use_direction, nbin, bin_option, flexible_alpha)
            finish = time.time()
            print('unmixing time: {} s'.format(finish - start))
            Table_time[ind_Exp, ind_alpha] = finish-start

            # Save the unmixed traces into a ".mat" file under folder "dir_trace_unmix".
            dir_trace_unmix = os.path.join(dir_traces, "alpha={:6.3f}".format(alpha))
            if not os.path.exists(dir_trace_unmix):
                os.makedirs(dir_trace_unmix)        
            savemat(os.path.join(dir_trace_unmix, Exp_ID+".mat"), {"traces_nmfdemix": traces_nmfdemix,\
                "list_mixout":list_mixout, "list_MSE":list_MSE, "list_final_alpha":list_final_alpha, "list_n_iter":list_n_iter})

        shm_video.close()
        shm_video.unlink()
        shm_masks.close()
        shm_masks.unlink()

        savemat(os.path.join(dir_traces, "Table_time.mat"), {"Table_time": Table_time, 'list_alpha': list_alpha})

