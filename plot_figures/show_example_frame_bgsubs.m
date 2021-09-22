dir_video='D:\ABO\20 percent 200\';
varname = '/mov';
dir_video_raw = dir_video;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% leng = 200;
%%         
eid = 10;
Exp_ID = list_Exp_ID{eid};
tic;
video_raw=h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
ROIs = FinalMasks;
toc;
[Lx,Ly,T] = size(video_raw);

% %%
% cmax = max(max(max(video_adjust)));
% cmin = min(min(min(video_adjust)));
% cmax_show = (cmax-cmin)/4+cmin;

% %% Calculate traces and background traces
tic;
% traces=generate_traces_from_masks(video_SNR,ROIs);
% [bgtraces,traces]=generate_bgtraces_from_masks(video_raw,ROIs);
[traces,bgtraces_mean,bgtraces_median]=generate_bgtraces_from_masks_3(video_raw,ROIs);
toc;
% load([dir_video_raw,'traces_ours_Raw_novideounmix\raw\',Exp_ID,'.mat'],...
%     'traces','bgtraces','outtraces');
% traces = traces';
% bgtraces = bgtraces';

%%
nn=16;
mask=ROIs(:,:,nn);
[xxs,yys]=find(mask>0);
xmin=min(xxs);%+floor((Lx-Lxm)/2);
xmax=max(xxs);%+floor((Lx-Lxm)/2);
ymin=min(yys);%+floor((Ly-Lym)/2);
ymax=max(yys);%+floor((Ly-Lym)/2);
xmin_show=max(1,2*xmin-xmax);
xmax_show=min(Lx,2*xmax-xmin);
ymin_show=max(1,2*ymin-ymax);
ymax_show=min(Ly,2*ymax-ymin);
trace_show=traces(nn,:);
% mask_ex=zeros(Lx,Ly);
% mask_ex(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:)=mask;
mask_show=mask(xmin_show:xmax_show,ymin_show:ymax_show);
% L_reshape=numel(mask_show);
% mask_show_reshape=reshape(mask_show,L_reshape,1);
range_d=[-1,3];

figure('Position',[480,550,800,250]); 
plot(trace_show);
xlabel('Time (frame)');
ylabel('F');
set(gca,'FontSize',12);
title(['Neuron ',num2str(nn),', Video ',Exp_ID]);

% %%
video_show=int16(video_raw(xmin_show:xmax_show,ymin_show:ymax_show,:));
mu_video = median(video_show,3);
AbsoluteDeviation = abs(video_show - mu_video); 
sigma_video = median(AbsoluteDeviation,3)/(sqrt(2)*erfinv(1/2)); 
% sigma_video_filt = std(video_filt_fd_show,1,3);
imshow_range=mean(mean(mu_video))+mean(mean(sigma_video))*range_d;
% imshow_range_filt_fd=mu_trace_filt_fd(nn)+sigma_trace_filt_fd(nn)*range_d;
% video_show=video_show(:,:,tmin_show:tmax_show);

% figure; imshow3D(video_show,imshow_range,6764);

% video_show=video_filt(xmin_show:xmax_show,ymin_show:ymax_show,:);
% imshow_range=[mu_video_filt(nn)-2*sigma_trace_filt(nn),mu_trace_filt(nn)+10*sigma_trace_filt(nn)];
% % video_show=video_SNR(xmin_show:xmax_show,ymin_show:ymax_show,:);
% % imshow_range=[mu_trace_SNR(nn)-2*sigma_trace_SNR(nn),mu_trace_SNR(nn)+10*sigma_trace_SNR(nn)];
% figure; imshow3D(video_show,imshow_range,t_max);

%%
imshow_range = [250,600];
t_input=12410;
% t_max_dFF=find(dFF==max(dFF));
% t_max=find(trace_show==max(trace_show))+t_max_dFF-1;
[~,t_max] = max(trace_show);
if ~exist('t_input','var') || t_input<=0
    t_show=t_max;
else
    t_show=t_input;
end
tmin_show = max(1,t_show-30);
tmax_show = min(T,t_show+30);
t_show0 = t_show-tmin_show+1;
image = video_show(:,:,t_show);
% %%
figure('Position',[480,550,480,400]);
imagesc(image,imshow_range); 
colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
hold on; contour(mask_show,'r');
str_title = ['Neuron ',num2str(nn),', Frame ',num2str(t_show)];
title(str_title);
set(gca,'FontSize',12);
saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

figure; imshow(image,imshow_range); 
% colorbar; 
hold on; contour(mask_show,'r');
% title(str_title);
mag=2;
mag_kernel = ones(mag,mag,'uint8');
crop_png=[86,64,size(image,2),size(image,1)];
img_all=getframe(gcf,crop_png);
cdata=img_all.cdata;
cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
for kc=1:3
    cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
end
imwrite(cdata_mag,[str_title,'.tif']);

%%
% figure; 
% plot(trace_show);
% % xlim([0,3000]);
% xlabel('Time (frame)');
% ylabel('F');
% set(gca,'FontSize',12);
% str_title = ['Neuron ',num2str(nn),', Video ',Exp_ID];
% title(str_title);
% saveas(gcf,['Neuron ',num2str(nn),', Video ',Exp_ID,'.png']);
