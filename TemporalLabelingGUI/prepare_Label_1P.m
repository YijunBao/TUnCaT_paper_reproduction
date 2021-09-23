clear;
addpath(genpath(pwd));

%% choose video file
dir_video='E:\OnePhoton videos\cropped videos\';
% varname = '/mov';
% dir_video_SNR = dir_video;
varname = '/network_input';
dir_video_SNR = fullfile(dir_video, 'SNR Video\');
dir_masks = fullfile(dir_video,'GT Masks\');
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
%%         
eid = 8;
Exp_ID = list_Exp_ID{eid};
tic;
video_SNR=h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),varname);
load([dir_video,'GT Masks merge\FinalMasks_',Exp_ID,'.mat'],'FinalMasks');
ROIs = FinalMasks;
toc;

%% Calculate traces and background traces
tic;
[bgtraces,traces]=generate_bgtraces_from_masks(video_SNR,ROIs);
toc;

%% Gamma correction
tic;
video_max=prctile(video_SNR(:),99);
video_SNR(video_SNR>video_max)=video_max;
video_min=prctile(video_SNR(:),10);
video_SNR=video_SNR-video_min;
video_SNR(video_SNR<0)=0;
video_adjust=sqrt(single(video_SNR));
video_adjust_min=min(min(min(video_adjust)));
video_adjust_max=max(max(max(video_adjust)));
video_adjust=(video_adjust-video_adjust_min)/(video_adjust_max-video_adjust_min);
toc;

%% Save prepared data
% [Lx,Ly,T]=size(video_adjust);
save('video_adjust.mat','video_adjust','traces','bgtraces','ROIs','-v7.3'); %,'-append'
%%
clear video_SNR

%% Manual labeling. 
NeuronFilter(video_adjust, ROIs, [], traces-bgtraces);
% NeuronFilter(video_adjust, ROIs, output, traces-bgtraces);

