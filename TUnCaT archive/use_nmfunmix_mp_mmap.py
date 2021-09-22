import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from scipy.io import savemat
from nmfunmix1_mmap import nmfunmix1


def use_nmfunmix(trace, bgtraces, masks_shape, fp_video, video_dtype, \
        fp_masks, fp_xx, fp_yy, comx, comy, r_bg, alpha, nbin=1):
    '''trace is the mixed traces (sum of values in mask) in (n, T) format n is index of neurons
    bgtraces is the background traces
    B  is the movie (should make it in "double" format)
    FinalMasks are the masks in (x,y,n) format
    photons = 1 or 2 indicates whether the video is from one-photon or
     two-photon microscope. Default is 2.
    demix is the demixed traces
    Relative data are stored in "demixtest.mat".

    radius (r_bg) for finding nearby neurons this is found by taking the average of
    the neuron diamters assuming a circle shape and multiplying by 1.25 - 1.5
    comx and comy are the centroids of masks.
    '''

    # parameter setting
    bin_option = 'sum'  # 'mean'
    (n, Lx, Ly) = masks_shape
    # FinalMasks = np.ndarray(masks_shape, buffer=fp_masks.buf, dtype = 'bool')

    # comx = np.zeros(n)
    # comy = np.zeros(n)
    # for nn in range(n):
    #     [xxs, yys] = FinalMasks[nn].nonzero()
    #     comx[nn] = xxs.mean()
    #     comy[nn] = yys.mean()

    p = mp.Pool(mp.cpu_count()//2)
    results = p.starmap(nmfunmix1, [(i, trace, bgtraces[:,i], fp_video, (Lx,Ly), video_dtype, fp_masks, \
        fp_xx, fp_yy, alpha, comx, comy, r_bg, nbin, bin_option) for i in range(n)], chunksize=1)
    p.close()
    list_traceout = ([x[0].T for x in results])
    demix = np.concatenate([x[0:1] for x in list_traceout], 0).T
    list_mixout = ([x[1] for x in results])
    list_neighbors = ([x[2] for x in results])
    list_outtrace = ([x[3] for x in results])
    list_tempmixIDs = ([x[4] for x in results])
    list_subtraces = ([x[5] for x in results])
    list_omitmasks = ([x[6] for x in results])
    list_questionmasks = np.array([x[7] for x in results])
    list_alpha = np.array([x[8] for x in results])

    savemat('demixtest.mat', {'demix':demix, 'trace':trace, 'list_traceout':list_traceout,\
        'list_mixout':list_mixout, 'list_neighbors':list_neighbors, 'list_subtraces':list_subtraces, \
        'list_outtrace':list_outtrace, 'list_alpha':list_alpha, 'list_tempmixIDs':list_tempmixIDs,\
        'list_omitmasks':list_omitmasks, 'list_questionmasks':list_questionmasks})

    # use_nmfunmix_refine('demixtest.mat')

    return demix
