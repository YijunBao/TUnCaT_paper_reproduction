color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
green = [0.1, 0.9, 0.1];
red = [0.9, 0.1, 0.1];
orange = color(2,:);
cyan = [0.1, 0.8, 0.9];
blue = [0.0, 0.4, 0.8];
yellow = [0.9, 0.9, 0.1];
magenta = [0.9, 0.3, 0.9];
purple = [0.7, 0.5, 0.7]; 
colors = distinguishable_colors(16);
colors(1:5,:)=color([3,2,6,5,4],:);
% colors(1:5,:)=color([3,6,1,5,4],:);

%%
dir_video='E:\OnePhoton videos\cropped videos\';
% varname = '/mov';
% dir_video_raw = dir_video;
varname = '/network_input';
dir_video_raw = fullfile(dir_video, 'SNR video\');
dir_masks = fullfile(dir_video,'GT Masks merge');
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% leng = 200;
%%         
eid = 9;
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
% tic;
% % traces=generate_traces_from_masks(video_SNR,ROIs);
% [bgtraces,traces]=generate_bgtraces_from_masks(video_raw,ROIs);
% toc;
load([dir_video,'traces_ours_SNR_merge_novideounmix_r2\raw\',Exp_ID,'.mat'],'traces','bgtraces');
traces = traces';
bgtraces = bgtraces';

%%
nn=1;
mask=ROIs(:,:,nn);
neighbors = find(sum(sum(ROIs.*mask)));
neighbors = setdiff(neighbors,nn)';

[xx, yy] = meshgrid(1:Ly,1:Lx); 
r_bg0=sqrt(mean(sum(sum(ROIs)))/pi)*2.5;
[xxs,yys]=find(mask>0);
comx=mean(xxs);
comy=mean(yys);
r_bg=max(r_bg0,sqrt(length(xxs)/pi)*2.5);
circle = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
half_show = 10;
xmin_show=min(Lx-2*half_show+1,max(1,ceil(comx-half_show)));
xmax_show=max(2*half_show,min(Lx,floor(comx+half_show)));
ymin_show=min(Lx-2*half_show+1,max(1,ceil(comy-half_show)));
ymax_show=max(2*half_show,min(Ly,floor(comy+half_show)));

% [xxs,yys]=find(mask>0);
% xmin=min(xxs);%+floor((Lx-Lxm)/2);
% xmax=max(xxs);%+floor((Lx-Lxm)/2);
% ymin=min(yys);%+floor((Ly-Lym)/2);
% ymax=max(yys);%+floor((Ly-Lym)/2);
% xmin_show=max(1,2*xmin-xmax);
% xmax_show=min(Lx,2*xmax-xmin);
% ymin_show=max(1,2*ymin-ymax);
% ymax_show=min(Ly,2*ymax-ymin);
trace_show=traces(nn,:); % -bgtraces(nn,:)
% mask_ex=zeros(Lx,Ly);
% mask_ex(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:)=mask;
mask_show=mask(xmin_show:xmax_show,ymin_show:ymax_show);
% L_reshape=numel(mask_show);
% mask_show_reshape=reshape(mask_show,L_reshape,1);
range_d=[-1,3]*3;

figure('Position',[480,550,800,250]); 
plot(trace_show);
xlabel('Time (frame)');
ylabel('F');
set(gca,'FontSize',12);
title(['Neuron ',num2str(nn),', Video ',Exp_ID],'Interpreter','None');

% %%
video_show=(video_raw(xmin_show:xmax_show,ymin_show:ymax_show,:));
mu_video = median(video_show,3);
AbsoluteDeviation = abs(video_show - mu_video); 
sigma_video = median(AbsoluteDeviation,3)/(sqrt(2)*erfinv(1/2)); 
% sigma_video_filt = std(video_filt_fd_show,1,3);
imshow_range=mean(mean(mu_video))+mean(mean(sigma_video))*range_d;
% imshow_range_filt_fd=mu_trace_filt_fd(nn)+sigma_trace_filt_fd(nn)*range_d;
% video_show=video_show(:,:,tmin_show:tmax_show);

% figure; imshow3D(video_show,imshow_range,3069);

% video_show=video_filt(xmin_show:xmax_show,ymin_show:ymax_show,:);
% imshow_range=[mu_video_filt(nn)-2*sigma_trace_filt(nn),mu_trace_filt(nn)+10*sigma_trace_filt(nn)];
% % video_show=video_SNR(xmin_show:xmax_show,ymin_show:ymax_show,:);
% % imshow_range=[mu_trace_SNR(nn)-2*sigma_trace_SNR(nn),mu_trace_SNR(nn)+10*sigma_trace_SNR(nn)];
% figure; imshow3D(video_show,imshow_range,t_max);

%%
% neighbors = [];
imshow_range = [-4,12]; % [250,500]; % [250,600]; % 
% list_t=[212, 697, 2407, 2754, 3458, 3697, 4551]; % 26 of 1
% list_t=[1559, 3069, 5024]; % 22 of 2
% list_t=[980, 1930, 3138, 4116, 4659, 4975]; % 28 of 3
list_t=[194, 1968, 3153, 4128, 5025]; % 1 of 9
% list_t=[1488, 1756, 3050, 3154]; % 2 of 1
% list_t=[112, 585, 1732, 2165, 2645, 3042, 3963, 4426, 5315, 2379, 2793, 3535]; % 22 of 4
% list_t=[379, 558, 4191, 4396]; % 2 of 7
% list_t=[194]; % 1 of 9
for t_input = list_t
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
[Lx2,Ly2]=size(image);
% %%
figure('Position',[480,550,480,400]);
imagesc(image,imshow_range); 
colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','SNR'); % Intensity
set(h,'FontSize',14);
hold on; contour(mask_show,'LineColor',blue,'LineWidth',2);
for n2 = 1:length(neighbors)
    neighbor = neighbors(n2);
    contour(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),'LineColor',colors(n2,:));
end
str_title = ['Neuron ',num2str(nn),', Frame ',num2str(t_show)];
title(str_title);
set(gca,'FontSize',12);
saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

figure; 
mag=4;
mag_kernel = ones(mag,mag,class(image));
imshow(kron(image,mag_kernel),imshow_range); 
% colorbar; 
hold on; contour(kron(mask_show,mag_kernel),'LineColor',blue,'LineWidth',2);
for n2 = 1:length(neighbors)
    neighbor = neighbors(n2);
    contour(kron(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),mag_kernel),'LineColor',colors(n2,:));
end
% title(str_title);
crop_png=[86,64,size(image,2)*mag,size(image,1)*mag];
img_all=getframe(gcf,crop_png);
cdata_mag=img_all.cdata;

if t_input == list_t(1)
    scale_bar = zeros(2,13,3,'uint8');
    scale_bar(:,:,2)=255;
    cdata_mag(5:6,end-16:end-4,:) = scale_bar;
end

% cdata_mag(mag*(1)+1:mag*(2),mag*1+1:mag*5,:) = scale_bar;
% cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
% for kc=1:3
%     cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
% end
imwrite(cdata_mag,[str_title,'.tif']);
end

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
