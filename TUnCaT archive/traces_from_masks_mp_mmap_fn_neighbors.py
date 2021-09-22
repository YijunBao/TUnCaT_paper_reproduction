import os
import numpy as np
import multiprocessing as mp


def mean_median_out_trace(nn, fn_video, video_dtype, video_shape, \
    fn_masks, fn_xx, fn_yy, r_bg, comx, comy, area, neighbors):
    (T, Lx, Ly) = video_shape
    n = comx.size
    fp_video = np.memmap(fn_video, dtype=video_dtype, mode='r', shape=(T,Lx*Ly))
    fp_masks = np.memmap(fn_masks, dtype='bool', mode='r', shape=(n,Lx,Ly))
    mask = fp_masks[nn]
    fp_xx = np.memmap(fn_xx, dtype='uint16', mode='r', shape=(Lx,Ly))
    fp_yy = np.memmap(fn_yy, dtype='uint16', mode='r', shape=(Lx,Ly))

    r_bg_0 = max(r_bg, np.sqrt(area/np.pi) * 2.5)
    circleout = (fp_yy-comx[nn]) ** 2 + (fp_xx-comy[nn]) ** 2 < r_bg_0 ** 2 # Background mask
    bgtrace = np.median(fp_video[:, circleout.ravel()], 1) # Background trace
    trace = fp_video[:, mask.ravel()].mean(1) # Neuron trace

    fgmask = fp_masks[neighbors].sum(0) > 0
    r_bg_large = r_bg * 0.8
    circleout = (fp_yy-comx[nn]) ** 2 + (fp_xx-comy[nn]) ** 2 < r_bg_large ** 2
    bgmask = np.logical_and(circleout,np.logical_not(fgmask)) # Outside mask
    bgweight = bgmask.sum() # Area of outside mask
    # Adjust the radius of the outside mask, so that the area of the outside mask is larger than half of the neuron area
    while bgweight < area/2: # == 0: # 
        r_bg_large = r_bg_large+1
        circleout = (fp_yy-comx[nn]) ** 2 + (fp_xx-comy[nn]) ** 2 < r_bg_large ** 2
        bgmask = np.logical_and(circleout,np.logical_not(fgmask))
        bgweight = bgmask.sum()
    outtrace = fp_video[:, bgmask.ravel()].mean(1) # Outside trace

    return trace, bgtrace, outtrace


def traces_bgtraces_from_masks_mmap_neighbors(fn_video, video_dtype, video_shape, \
        fn_masks, masks_shape, fpasks, name_mmap):
    (ncells, Lx, Ly) = masks_shape
    (list_neighbors, comx, comy, area, r_bg) = find_neighbors(fpasks)

    [xx, yy] = np.meshgrid(np.arange(Ly), np.arange(Lx))
    xx = xx.astype('uint16')
    fn_xx = name_mmap + 'xx.dat'
    fp_xx = np.memmap(fn_xx, dtype='uint16', mode='w+', shape=(Lx,Ly))
    fp_xx[:] = xx[:]
    yy = yy.astype('uint16')
    fn_yy = name_mmap + 'yy.dat'
    fp_yy = np.memmap(fn_yy, dtype='uint16', mode='w+', shape=(Lx,Ly))
    fp_yy[:] = yy[:]

    p = mp.Pool(mp.cpu_count())
    results = p.starmap(mean_median_out_trace, [(nn, fn_video, video_dtype, video_shape, \
        fn_masks, fn_xx, fn_yy, r_bg, comx, comy, area[nn], list_neighbors[nn]) for nn in range(ncells)], chunksize=1)
    p.close()

    traces = np.vstack([x[0] for x in results]).T
    bgtraces = np.vstack([x[1] for x in results]).T
    outtraces = np.array([x[2] for x in results]).T

    fp_xx._mmap.close()
    del fp_xx
    os.remove(fn_xx)
    fp_yy._mmap.close()
    del fp_yy
    os.remove(fn_yy)

    return traces, bgtraces, outtraces, list_neighbors


def find_neighbors(FinalMasks):
    ''' Find the neighboring neurons of all neurons. 
    Inputs: 
        FinalMasks (numpy.ndarray of bool, shape = (n,Lx,Ly)): The binary spatial masks of all neurons.
            Each slice represents a binary mask of a neuron.

    Outputs:
        list_neighbors (list of list of int): 
            Each element is a list of indeces of neighboring neurons of a neuron.
        comx (numpy.ndarray of float, shape = (n,)): The x-position of the center of each mask.
        comy (numpy.ndarray of float, shape = (n,)): The y-position of the center of each mask.
        area (numpy.ndarray of int, shape = (n,)): The area of each mask.
        r_bg (float): The radius of the background mask.
    '''

    # (n, Lx, Ly) = FinalMasks.shape
    n = FinalMasks.shape[0]
    list_neighbors = []
    area = np.zeros(n, dtype = 'uint32')
    comx = np.zeros(n)
    comy = np.zeros(n)
    for i in range(n):
        mask = FinalMasks[i]
        [xxs, yys] = mask.nonzero()
        area[i] = xxs.size
        comx[i] = xxs.mean()
        comy[i] = yys.mean()

    r_bg = np.sqrt(area.mean()/np.pi)*2.5
    for i in range(n):
        # The distance between each pair of neurons
        r = np.sqrt((comx[i]-comx) **2 + (comy[i]-comy) **2)
        # Neighboring neurons are defined as the neurons whose centers are 
        # less than "r_bg" away from the center of the neuron of interest. 
        neighbors = np.concatenate([np.array([i]), np.logical_and(r > 0, r < r_bg).nonzero()[0]])
        list_neighbors.append(neighbors)

    return list_neighbors, comx, comy, area, r_bg

