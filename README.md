![NeuroToolbox logo](readme/neurotoolbox-logo.svg)

# TUnCaT_paper_reproduction
Data and reproduction code for the TUnCaT paper.

Copyright (C) 2021 Duke University NeuroToolbox.

Detailed folder structure of this repository is shown in `folder_structure.txt`. We give a general introduction of the files of this repository below. All the data to reproduce the main figures, including the videos, masks, manual labels, unmixed traces, processing time, and F1 scores are provided. Videos and unmixed traces to reproduce the supplementary figures are not provided because they are too large, but the processing time and F1 scores to reproduce the supplementary figures are provided. Most of the code used in this paper is provided, except the implementation of NAOMi. If you want to get the data or code not shared in this repository, please email us. 


# Videos and GT labels
In our paper, we used experimental two-photon imaging videos from Allen Brain Observatory (ABO) dataset, simulated two-photon imaging videos using NAOMi, and experimental one-photon imaging videos from our lab. We used the manually labeled GT masks and GT transients for experimental videos, and used the automatically generated GT masks and GT transients for simulated videos. A more detailed instruction is given below.


## Allen Brain Observatory (ABO) dataset
The ABO dataset is available in [Allen Institute](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS). You may need a Amazon AWS account to download them. We used ten videos from 275 um layer, {'524691284', '531006860', '502608215', '503109347', '501484643', '501574836', '501729039', '539670003', '510214538', '527048992'}. We used the manually labeled masks of [275 um layer](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO/Layer275/FinalGT) created by Soltanian-Zadeh et al. We also used the code [create_h5_video_ABO.m](data/ABO/create_h5_video_ABO.m) modified from the same STNeuroNet repository to crop each video to the first 20% durations and the center 200 x 200 pixels, so that the video sizes are changed from 512 pixels x 512 pixels x ~115,000 frames to 200 pixels x 200 pixels x ~23,000 frames. We also matched the spatial cropping to the GT masks, which are provided in "`data/ABO/GT Masks`". We provided the manually labeled GT transients in "`data/ABO/GT transients`"


## NAOMi dataset
We implemeted NAOMi using the available [MATLAB code](https://codeocean.com/capsule/7031153/tree/v1) (version 1.1) to simulate two-photon videos. We changed the indicator concentration from 200 μM to 10 μM to be more realistic. Please refer to our paper for detailed parameter settings. We provided the videos used in Fig 3 in `data/NAOMi/videos`. We provided the corresponding GT masks and GT transients in `data/NAOMi/GT Masks`. 


## One-photon dataset
The one-photon videos were recorded in our lab. We provided the videos in `data/1p/videos`. We provided GT masks in `data/1p/GT Masks`. We provided the manually labeled GT transients in `data/1p/GT transients`


## Generating SNR videos (Python)
In addition to unmix the traces from raw videos, we also unmixed the traces for SNR videos. We used the code in the folder `SNR convertion` to convert all the raw videos to SNR videos. Required packages: (existing) numpy, scipy, h5py, numba, (additional) pyfftw, cpuinfo, opencv. Follow `bat_SNR.bat` to run the code. The template used in temporal filtering are provided in the folder `template`. 


# Manually label GT transients (MATLAB)
We manually labeled the GT transients for experimental videos using a MATLAB GUI in the folder `TemporalLabelingGUI`. We used `prepare_Label_ABO.m` to label ABO videos and used `prepare_Label_1P.m` to label one-photon vieos. 

<img src="readme/manual labeling GUI.png" height="500"/>

*Select neuron and potential transient.* 
A typical GUI window is shown above. The two lists on the top right corner are the list of neurons (top) and the list of potential transients of that neuron (bottom). When double clicking the numbers in the lists, that neuron or transient will be displayed. The buttons "Forward Neuron" and "Backward Neuron" can display the neuron with one index larger or smaller than the current neuron. 

*Display panels.*
The "Mask" panel shows the masks of the neuron of interest (red) and neighboring neurons (other colors), overlaid on the maximum projection image. The "Trace" panel shows the SNR trace of the selected neuron; the green dashed horizontal line shows the SNR threshold; the rectangles show periods higher than the SNR threshold; the trangles show prominant peaks. Squares overlaid with trangles show potential transients to be labeled, and these potential transients are listed in the second list on the top right. The "Video" panel plays the entire video after clicking "Play Video", or plays a short video from 20 frames before the start of the transient to 20 frames after the end of the transient after clicking "Select Spike", "Yes Spike", "No Spike", or "Replay Spike Video". When the video is playing, a red vertical line is moving in the "Trace" panel along the video frames. After the video is played, the "Video" panel will display the mean image over the transient. 

*Label potential transients as true or false.*
There are eight buttons on the bottom right. "Play Video" will play the entire video. "Yes Active" and "No Active" are currently not used. "Save Changes" saves the labels to `TemporalLabelingGUI/output/output{}.mat`. "Select Spike" will start labeling the spikes of this neuron, and it will play the video of the first transient. "Yes Spike" will label this transient as a true transient and move to the next transient. "No Spike" will label this transient as a false transient and move to the next transient. "Replay Spike Video" will replay this transient. 


# TUnCaT code (Python)
The version of TUnCaT used in this paper is provided in the folder `TUnCaT archive`. Required packages: numpy, scipy, h5py, scikit-learn, numba. A demo is provided. 

The wrapper scripts to process all the test datasets are stores in the same folder. Python 3.8 or newer is required to run the wrapper scripts. Run `run_TUnCaT_all.bat` to run all the wrapper scripts. The unmixed traces will be saved in `data/{data}/traces_ours_{}`, which should be nearly the same as the provided unmixed traces in `results/{data}/unmixed traces/traces_ours_{}`.


# Other unmixing methods (Python)
In our paper, we compared the performance of TUnCaT with three peer temporal unmixing methods: [FISSA](https://github.com/rochefort-lab/fissa) (version 0.7.2), [CNMF](https://github.com/flatironinstitute/CaImAn) (version 1.6.4), and the [Allen SDK](https://github.com/AllenInstitute/AllenSDK) (version 2.7.0). Follow the instructions of these algorithms on their official websites to install them. We provided the wrapper scripts to run these methods on all datasets in the foler `other methods`. We also provided .bat files to include all the running commands for each algorithm. The unmixed traces will be saved in `data/{data}/traces_{method}_{}`, which should be nearly the same as the provided unmixed traces in `results/{data}/unmixed traces/traces_{method}_{}`. 

In addition, we also used spatial segmentation method [SUNS](https://github.com/YijunBao/Shallow-UNet-Neuron-Segmentation_SUNS) (version 1.1.1) and spatiotemporal unmixing method [CaImAn](https://github.com/flatironinstitute/CaImAn) (version 1.6.4) to test the pipeline containing both spatial neuron segmentation and temporal trace unmixing on ABO dataset. Follow the instructions of these algorithms on their official websites to install them. We tested SUNS+TUnCaT, SUNS+CNMF, and CaImAn. We provided the wrapper scripts of SUNS, SUNS+CNMF, and CaImAn in the foler `other methods`. We provided the wrapper scripts of SUNS+TUnCaT in the foler `TUnCaT archive`. For ease of testing SUNS+TUnCaT and SUNS+CNMF, we also provided the spatial masks generated by SUNS in the folder `data/ABO/SUNS_complete Masks`, so that these test can be run without running SUNS. 


# Evaluation code (MATLAB)
We evaluated the accuracy of the unmixed traces by calculating the F1 scores through cross-valitaion. The code is provided in the folder `evaluation`. 
The main evaluation scripts are `eval_{data}_{method}.m`, and users can use `eval_all.m` to run all the evaluation code together. The resulting F1 scores will saved in `evaluation/{data}`, which should be nearly the same as the provided F1 scores in `results/{data}/evaluation`. By default, the code will process the unmixed traces provided in `results/{data}/unmixed traces`, but users can change the folders to process the unmixed traces they generated in `data/{data}` by changing `dir_traces`.
The output scores and processing time can be summarized by `summary_F1_{}.m`, and users can use `summary_all.m` to run all the summary code together. The resulting summary will saved in `evaluation/{data}`, which should be nearly the same as the provided summary in `results/{data}/evaluation`. By default, the code will process the performance measurement provided in `results/{data}/evaluation` and the processing time provides in `results/{data}/unmixed traces`, but users can change the folders to process the performance measurement they generated in `evaluation/{data}` and the processing time provides in `data/{data}` by changing `dir_scores` and `dir_traces`. 
However, `eval_{data}_ours_bin.m` can only process the users-generated traces, because we did not provide our traces due to their large sizes. Some other scripts can only process our provided data, also because we did not provide our raw data due to their large sizes.


# Unmixing results and F1 scores
The unmixed traces of all unmixing methods on both raw videos and SNR videos are provided in `results/{data}/unmixed traces`. For FISSA and TUnCaT (ours), we provided the unmixed traces using different alpha, as well as the raw traces before NMF (equivalent to alpha = 0). Other methods have only one set of output traces. The processing times are provided in `results/{data}/unmixed traces/traces_{method}_{video format}/Table_time.mat`. For TUnCaT with temporal downsampling, we only provided the processing time, because the traces are too large. The F1 scores are provided in `results/{data}/evaluation`. 


# Plot figures in the paper
The code used to plot the main figures and supplementary figures in the paper is provided in `plot figures`. Users can run `plot_figures_all.m` to run all the cdoe together. By default, the code will use the unmixed traces and performance measurement provided in `results/{data}`, but users can change the folders to use the unmixed traces they generated in `data/{data}` and use the performance measurement they generated in `evaluation/{data}`. However, Figs S4, S6 and S7 cannot be regenerated from the raw videos, because we did not provide the large raw videos.


# Licensing and Copyright
TUnCaT is currently private, but it will be released under [the GNU License, Version 2.0](LICENSE) after paper acceptance.


# Sponsors
<img src="readme/NSFBRAIN.png" height="100"/><img src="readme/BRF.png" height="100"/><img src="readme/Beckmanlogo.png" height="100"/>
<br>
<img src="readme/valleelogo.png" height="100"/><img src="readme/dibslogo.png" height="100"/><img src="readme/sloan_logo_new.jpg" height="100"/>
