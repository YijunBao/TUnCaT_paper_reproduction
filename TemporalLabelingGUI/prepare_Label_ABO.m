addpath(genpath('C:\Matlab Files\Unmixing'));
addpath(genpath('C:\Matlab Files\Filter'));
addpath(genpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools'));

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
% leng = 200;
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
save('video_adjust.mat','video_adjust','traces','bgtraces','ROIs','-v7.3'); %,'-append'
%%
clear video_SNR
%%
NeuronFilter(video_adjust, ROIs, [], traces-bgtraces);
% NeuronFilter(video_adjust, ROIs, output, traces-bgtraces);
output;
%%
nn=7;
mask = ROIs(:,:,nn);
[xx,yy] = find(mask);
[lx,ly] = size(mask);
xmin = min(xx);
xmax = max(xx);
ymin = min(yy);
ymax = max(yy);
xrange = max(xmin-5,1):min(xmax+5,lx);
yrange = max(ymin-5,1):min(ymax+5,ly);
figure('Position',[100,500,300,300]); 
imagesc(ROIs(xrange, yrange,nn)); axis('image','off');
figure('Position',[500,500,300,300]); 
% imshow3D(video_adjust(xrange, yrange,:),[],[2154]);
cmax = max(max(max(video_adjust)));
cmin = min(min(min(video_adjust)));
cmax_show = (cmax-cmin)/4+cmin;
imshow3D(video_adjust(xrange, yrange,:),[cmin, cmax_show],[2154]);
% figure('Position',[100,500,900,400]); 
% subplot(1,2,1)
% imagesc(ROIs(xrange, yrange,nn)); axis('image');
% subplot(1,2,2)
% imshow3D(video_adjust(xrange, yrange,:),[],585);

% plot_masks_id(ROIs);
%%
% figure; imagesc(ROIs(:,:,300)); axis('image');
% figure; imshow3D(video_adjust(251:322,437:487,:),[],12044);
% 501574836: remove neuron 11
for nn = 1:length(output)
    output_now = output{nn};
    if ~isempty(output_now)
        dist = output_now(2:end,1) - output_now(1:end-1,2);
        close = find(dist<=2); 
        if ~isempty(close)
            fprintf('%d: %s\n',nn, mat2str(close'));
        end
    end
end
