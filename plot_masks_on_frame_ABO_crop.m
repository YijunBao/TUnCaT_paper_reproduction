color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %
colors_multi = distinguishable_colors(16);

save_figures = true;
mag=4;
mag_kernel = ones(mag,mag,'uint8');
SNR_range = [2,10];
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% neurons and masks frame
dir_video='D:\ABO\20 percent 200\';
% varname = '/mov';
dir_video_raw = dir_video;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_GT_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% DirData = 'D:\ABO\';
% dir_raw = 'D:\ABO\20 percent\';
dir_video_SNR = [dir_video,'SNR video\'];
% dir_traces = 'D:\ABO\20 percent\complete\traces\';

k=1;
xrange=1:200; yrange=1:200;
% xrange = 334:487; yrange=132:487;
% xrange = 1:487; yrange=1:487;
crop_png=[86,64,length(yrange),length(xrange)];
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
% load(['ABO mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
% load(['ABO mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
% unmixed_traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
video_raw = h5read([dir_video_raw,Exp_ID,'.h5'],'/mov'); % raw_traces
raw_max = max(video_raw,[],3);
video_SNR = h5read([dir_video_SNR,Exp_ID,'.h5'],'/network_input'); % raw_traces
SNR_max = max(video_SNR,[],3);

load([dir_GT_masks, '\FinalMasks_', Exp_ID, '.mat'], 'FinalMasks');
GT_Masks = logical(FinalMasks);
% FinalMasks = permute(FinalMasks,[2,1,3]);
GT_Masks_sum = sum(GT_Masks,3);

%% SUNS 
dir_output_mask = [dir_video,'SUNS_complete Masks\FinalMasks_'];
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load([dir_output_mask, Exp_ID, '.mat'], 'FinalMasks');
SUNS_Masks = FinalMasks; % permute(Masks,[3,2,1]);
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,SUNS_Masks,0.5);
% Masks = permute(Masks,[2,1,3]);
SUNS_Masks_sum = sum(SUNS_Masks,3);

TP_2 = sum(m,1)>0;
TP_22 = sum(m,2)>0;
FP_2 = sum(m,1)==0;
FN_2 = sum(m,2)==0;
masks_TP = sum(SUNS_Masks(:,:,TP_2),3);
masks_FP = sum(SUNS_Masks(:,:,FP_2),3);
masks_FN = sum(GT_Masks(:,:,FN_2),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[50,50,500,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imshow(SNR_max(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
contour(GT_Masks_sum(xrange,yrange), 'Color', color(3,:));
contour(SUNS_Masks_sum(xrange,yrange), 'Color', colors_multi(7,:));

title(sprintf('SUNS, F1 = %1.2f',F1),'FontSize',12);
% h=colorbar;
% set(get(h,'Label'),'String','Peak SNR');
% set(h,'FontSize',12);

% % rectangle('Position',rect1,'EdgeColor',yellow,'LineWidth',2);
% % rectangle('Position',rect2,'EdgeColor',yellow,'LineWidth',2);
% rectangle('Position',rect3,'EdgeColor',color(7,:),'LineWidth',2);
% % rectangle('Position',rect4,'EdgeColor',yellow,'LineWidth',2);
% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    saveas(gcf,[Exp_ID,' SNR SUNS.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% figure(1)
img_all=getframe(gcf,crop_png);
cdata=img_all.cdata;
% cdata=permute(cdata,[2,1,3]);
% figure; imshow(cdata);

if save_figures
    imwrite(permute(cdata,[2,1,3]),[Exp_ID,' SNR SUNS.tif']);
end


%% CaImAn Batch
dir_output_mask = [dir_video,'caiman-Batch_raw\Masks\'];
% dir_output_mask = 'D:\ABO\20 percent bin 5\CaImAn-Batch\Masks\';
load([dir_output_mask, Exp_ID, '_neurons.mat'], 'finalSegments');
CaImAn_Masks = finalSegments;
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,CaImAn_Masks,0.5);
% Masks = permute(Masks,[2,1,3]);
CaImAn_Masks_sum = sum(CaImAn_Masks,3);

TP_4 = sum(m,1)>0;
TP_42 = sum(m,2)>0;
FP_4 = sum(m,1)==0;
FN_4 = sum(m,2)==0;
masks_TP = sum(CaImAn_Masks(:,:,TP_4),3);
masks_FP = sum(CaImAn_Masks(:,:,FP_4),3);
masks_FN = sum(GT_Masks(:,:,FN_4),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,4)
figure('Position',[650,50,500,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imshow(SNR_max(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
contour(GT_Masks_sum(xrange,yrange), 'Color', color(3,:));
contour(CaImAn_Masks_sum(xrange,yrange), 'Color', color(4,:));

title(sprintf('CaImAn, F1 = %1.2f',F1),'FontSize',12);
% h=colorbar;
% set(get(h,'Label'),'String','Peak SNR');
% set(h,'FontSize',12);

if save_figures
    saveas(gcf,[Exp_ID,' SNR CaImAn.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% figure(1)
img_all=getframe(gcf,crop_png);
cdata=img_all.cdata;
% cdata=permute(cdata,[2,1,3]);
% figure; imshow(cdata);

if save_figures
    imwrite(permute(cdata,[2,1,3]),[Exp_ID,' SNR CaImAn.tif']);
end

%% Plot raw max
% load(['ABO mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
% figure
% imshow(raw_max(xrange,yrange),[300,1200]); axis('image'); colormap gray; %
% xticklabels({}); yticklabels({});
% hold on;
% contour(FinalMasks_sum(xrange,yrange), 'Color', green); % green
% % colorbar;
% title('Raw max');
% % rectangle('Position',rect1,'EdgeColor',magenta,'LineWidth',2);
% % rectangle('Position',rect2,'EdgeColor',magenta,'LineWidth',2);
% % rectangle('Position',rect3,'EdgeColor',magenta,'LineWidth',2);
% % rectangle('Position',rect4,'EdgeColor',magenta,'LineWidth',2);
% rectangle('Position',rect5,'EdgeColor',magenta,'LineWidth',2);

%%
figure('Position',[1250,750,500,300],'Color','w');
imshow(SNR_max(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','Peak SNR','FontName','Arial');
set(h,'FontSize',12);
% if save_figures
%     saveas(gcf,'colorbar_SNR.svg');
% %     save(['trace\',Exp_ID,' N',num2str(N_neuron),' trace.mat'],'trace_N');
% end
