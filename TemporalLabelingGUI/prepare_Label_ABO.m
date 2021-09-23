clear;
addpath(genpath(pwd));

%% choose video file
clear;
dir_video='D:\ABO\20 percent 200\';
% varname = '/mov';
% dir_video_SNR = dir_video;
varname = '/network_input';
dir_video_SNR = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
%%         
eid = 10;
Exp_ID = list_Exp_ID{eid};
tic;
video_SNR=h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
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

