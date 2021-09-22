![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# TUnCaT_paper_reproduction
Data and reproduction code for the TUnCaT paper.

Copyright (C) 2021 Duke University NeuroToolbox.

Detailed folder structure of this repository is shown in `folder_structure.txt`. We give a general introduction of the files of this repository below. All the data to reproduce the main figures, including the videos, masks, manual labels, unmixed traces, processing time, and F1 scores are stored. Videos and unmixed traces to reproduce the supplementary figures are not stored because they are too large, but the processing time and F1 scores to reproduce the supplementary figures are stored. Most of the code used in this paper is stored, but the implementation of other open-source code, including NAOMi and other unmixing methods, is not stored. If you want to get the data or code not shared in this repository, please email us. 


# Videos and GT labels
In our paper, we used experimental two-photon imaging videos from Allen Brain Observatory (ABO) dataset, simulated two-photon imaging videos using NAOMi, and experimental one-photon imaging videos from our lab. We used the manually labeled GT masks and GT transients for experimental videos, and used the automatically generated GT masks and GT transients for simulated videos. A more detailed instruction is given below.


## Allen Brain Observatory (ABO) dataset
The ABO dataset is available in [Allen Institute](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS). You may need a Amazon AWS account to download them. We used ten videos from 275 um layer, {'524691284', '531006860', '502608215', '503109347', '501484643', '501574836', '501729039', '539670003', '510214538', '527048992'}. We used the manually labeled masks of [275 um layer](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO/Layer275/FinalGT) created by Soltanian-Zadeh et al. We also used the code [create_h5_video_ABO.m](data/ABO/create_h5_video_ABO.m) modified from the same STNeuroNet repository to crop each video to the first 20% durations and the center 200 x 200 pixels, so that the video sizes are changed from 512 x 512 x ~115,000 to 200 x 200 x ~23,000. We also matched the spatial cropping to the GT masks, which are stored in "`data/ABO/GT Masks`". We stored the manually labeled GT transients in "`data/ABO/GT transients`"


## NAOMi dataset
We implemeted NAOMi using the available [MATLAB code](https://codeocean.com/capsule/7031153/tree/v1) (version 1.1) to simulate two-photon videos. We changed the indicator concentration from 200 μM to 10 μM to be more realistic. Please refer to our paper for detailed parameter settings. We stored the videos used in Fig 3 in `data/NAOMi/videos`. We stored the corresponding GT masks and GT transients in `data/NAOMi/GT Masks`. 


## One-photon dataset
The one-photon videos were recorded in our lab. We stored the videos in `data/1p/videos`. We stored GT masks in `data/1p/GT Masks merge`. We stored the manually labeled GT transients in `data/1p/GT transients`


## Generating SNR videos (Python)
In addition to unmix the traces from raw videos, we also unmixed the traces for SNR videos. We used the code in `SNR convertion` to convert all the raw videos to SNR videos. Required packages: numpy, scipy, h5py, numba, pyfftw, cpuinfo, opencv. Follow `bat_SNR.bat` to run the code. The template used in temporal filtering are stored in `template`. 


# Manually label GT transients (MATLAB)
We manually labeled the GT transients for experimental videos using a MATLAB GUI in `TemporalLabelingGUI`. We used `prepare_Label_ABO.m` to label ABO videos and used `prepare_Label_1P.m` to label one-photon vieos. 

<img src="readme/example manual labeling GUI.png" height="500"/>

A typical GUI window is shown above. The two lists on the top right corner are the list of neurons (top) and the list of potential transients of that neuron (bottom). When double clicking the numbers in the lists, that neuron or transient will be displayed. The buttons "Forward Neuron" and "Backward Neuron" can display the neuron with one index larger or smaller than the current neuron. 
The "Mask" panel shows the masks of the neuron of interest (red) and neighboring neurons (other colors), overlaid on the maximum projection image. The "Trace" panel shows the SNR trace of the selected neuron; the green dashed horizontal line shows the SNR threshold; the rectangles show periods higher than the SNR threshold; the trangles show prominant peaks. Squares overlaid with trangles show potential transients to be labeled, and these potential transients are listed in the second list on the top right. The "Video" panel plays the entire video after clicking "Play Video", or plays a short video from 20 frames before the start of the transient to 20 frames after the end of the transient after clicking "Select Spike", "Yes Spike", "No Spike", or "Replay Spike Video". When the video is playing, a red vertical line is moving in the "Trace" panel along the video frames. After the video is played, the "Video" panel will display the mean image over the transient. 
There are eight buttons on the bottom right. "Play Video" will play the entire video. "Yes Active" and "No Active" are currently not used. "Save Changes" saves the labels to `TemporalLabelingGUI/output/output{}.mat`. "Select Spike" will start labeling the spikes of this neuron, and it will play the video of the first transient. "Yes Spike" will label this transient as a true transient and move to the next transient. "No Spike" will label this transient as a false transient and move to the next transient. "Replay Spike Video" will replay this transient. 


# TUnCaT code (Python)
The version of TUnCaT used in this paper and some calling scripts to process the videos are stored in `TUnCaT archive`. Currently python 3.8 or newer is required to use the SharedMemory module. Required packages: numpy, scipy, h5py, scikit-learn, numba. Follow `do_something.bat` to run the code. 


# Evaluation code (MATLAB)
We evaluated the accuracy of the unmixed traces by calculating the F1 scores through cross-valitaion. The code is stored in `evaluation`. The main scripts are `eval_{data}_{method}.m`, and the output scores can be summarized by `summary_timing_ABO_all.m`. The F1 scores through cross-valiation are stored in `results/{data}/evaluation`. 


# Unmixing results and F1 scores
The unmixed traces of all unmixing methods on both raw videos and SNR videos are stored in `results/{data}/unmixed traces`. For FISSA and TUnCaT (ours), we stored the unmixed traces using different alpha, as well as the raw traces before NMF (equivalent to alpha = 0). Other methods have only one set of output traces. The processing times are stored in `results/{data}/unmixed traces/traces_{method}_{video format}/Table_time.mat`. The F1 scores through cross-valiation are stored in `results/{data}/evaluation`. 


# Plot figures in the paper
The code used to plot the figures and SI figures in the paper is stored in `plot figures`. Detailed correspondance can be found in `folder_structure.txt`.


# Licensing and Copyright
TUnCaT is currently private, but it will be released under [the GNU License, Version 2.0](LICENSE) after paper acceptance.


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
