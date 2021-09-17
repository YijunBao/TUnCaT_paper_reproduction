% clear;
addpath(genpath('C:\Matlab Files\Unmixing'));
addpath(genpath('C:\Matlab Files\Filter'));
addpath(genpath('C:\Matlab Files\TemporalLabelingGUI-master'));
addpath(genpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools'));

%% choose video file
dir_video='E:\OnePhoton videos\cropped videos\';
% varname = '/mov';
% dir_video_SNR = dir_video;
varname = '/network_input';
dir_video_SNR = fullfile(dir_video, 'SNR Video\');
dir_masks = fullfile(dir_video,'GT Masks\');
% list_Exp_ID={'c25', 'c27', 'c28'};
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% leng = 200;
%%         
eid = 8;
Exp_ID = list_Exp_ID{eid};
tic;
video_SNR=h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),varname); % ,[1,1,1],[487,487,inf]
load([dir_video,'GT Masks merge\FinalMasks_',Exp_ID,'.mat'],'FinalMasks');
ROIs = FinalMasks;
toc;

%% Calculate traces and background traces
tic;
% traces=generate_traces_from_masks(video_SNR,ROIs);
[bgtraces,traces]=generate_bgtraces_from_masks(video_SNR,ROIs);
toc;
%%
% video_min=min(min(min(video_filt)));
% video_max=max(max(max(video_filt)));
% video_adjust=(video_filt-video_min)/(video_max-video_min);
% for ii = 1:size(video_filt, 3)
%     video_adjust(:, :, ii) = imadjust(video_adjust(:, :, ii), [], [], 0.5);
% end
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
%%
% [Lx,Ly,T]=size(video_adjust);
% save('video_adjust.mat','video_adjust','traces','bgtraces','ROIs','-v7.3'); %,'-append'
%%
% clear video_SNR
%%
NeuronFilter(video_adjust, ROIs, [], traces-bgtraces);
% NeuronFilter(video_adjust, ROIs, output, traces-bgtraces);
%%
% nn=42;
% mask = ROIs(:,:,nn);
% [xx,yy] = find(mask);
% [lx,ly] = size(mask);
% xmin = min(xx);
% xmax = max(xx);
% ymin = min(yy);
% ymax = max(yy);
% xrange = max(xmin-5,1):min(xmax+5,lx);
% yrange = max(ymin-5,1):min(ymax+5,ly);
% figure('Position',[100,500,300,300]); 
% imagesc(ROIs(xrange, yrange,nn)); axis('image','off');
% figure('Position',[500,500,300,300]); 
% imshow3D(video_adjust(xrange, yrange,:),[],[[[[[[4549]]]]]]);
% % figure('Position',[100,500,900,400]); 
% % subplot(1,2,1)
% % imagesc(ROIs(xrange, yrange,nn)); axis('image');
% % subplot(1,2,2)
% % imshow3D(video_adjust(xrange, yrange,:),[],585);
% % 501574836: remove neuron 11

% plot_masks_id(ROIs);
