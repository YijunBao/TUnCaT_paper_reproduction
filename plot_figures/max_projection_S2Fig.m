clear;
addpath(genpath('..\evaluation'));
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
yellow = color(3,:);
% yellow = [0.9, 0.9, 0.1];
magenta = [0.9, 0.3, 0.9];
purple = [0.7, 0.5, 0.7]; 
% colors_multi = distinguishable_colors(16);
% colors = [cyan; colors_multi(5,:)];
colors = [yellow; cyan];
% transparency = 0.7;
% addpath(genpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools'));

%% ABO 1
% dir_video='D:\ABO\20 percent 200\';
dir_video='..\data\ABO\';
% dir_trace='..\results\ABO\unmixed traces\';
varname = '/mov';
dir_video_raw = dir_video;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% leng = 200;
% %%         
eid = 10;
list_nn = [16,51]; 
% eid = 4;
% list_nn = [55]; 
Exp_ID = list_Exp_ID{eid};
tic;
video_raw=h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
ROIs = FinalMasks;
toc;
[Lx,Ly,T] = size(video_raw);
image_max = max(video_raw,[],3);
num_neuron = size(ROIs,3);

% %%
% imshow_range = [min(image_max,[],'all'),max(image_max,[],'all')/2]; % 
imshow_range = [000,2000];
figure('Position',[480,550,480,400]);
imshow(image_max,imshow_range);  % 
hold on;

for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    [xx, yy] = meshgrid(1:Ly,1:Lx); 
    r_bg0=sqrt(mean(sum(sum(ROIs)))/pi)*2.5;
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    r_bg=max(r_bg0,sqrt(length(xxs)/pi)*2.5);
    circle = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
    half_show = 20;
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

    rect_region = [ymin_show, xmin_show, ymax_show-ymin_show+1, xmax_show-xmin_show+1];
    rectangle('Position',rect_region,'EdgeColor',colors(ind,:),'LineWidth',1);
end

for n2 = 1:num_neuron
    if ~any(n2==list_nn)
        contour(ROIs(:,:,n2),'LineColor',orange);
%         edge_mask = edge(ROIs(:,:,n2))*transparency;
%         color_all = uint8(floor(ones(Lx,Ly,3).*reshape(cyan,[1,1,3])*256));
%         imagesc(color_all, 'AlphaData', edge_mask);
    end
end
for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    contour(mask,'LineColor',blue,'LineWidth',2);
%     edge_mask = edge(mask,'nothinning')*transparency;
%     color_all = uint8(floor(ones(Lx,Ly,3).*reshape(red,[1,1,3])*256));
%     imagesc(color_all, 'AlphaData', edge_mask);
end
str_title = [Exp_ID,' max'];
title(str_title);
set(gca,'FontSize',12);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

mag=1;
% figure; 
% mag_kernel = ones(mag,mag,class(image));
% imshow(kron(image,mag_kernel),imshow_range); 
% % colorbar; 
% hold on; contour(kron(mask_show,logical(mag_kernel)),'LineColor',red);
% for n2 = 1:length(neighbors)
%     neighbor = neighbors(n2);
%     contour(kron(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),logical(mag_kernel)),'LineColor',colors(n2,:));
% end
% % contour(kron(circle(xmin_show:xmax_show,ymin_show:ymax_show),logical(mag_kernel)),'LineColor',colors(2,:))

% title(str_title);
crop_png=[86,64,size(image_max,2)*mag,size(image_max,1)*mag];
img_all=getframe(gcf,crop_png);
cdata_mag=img_all.cdata;

% eid = 4;
scale_bar = zeros(2,13,3,'uint8');
% scale_bar(:,:,1)=128;
% scale_bar(:,:,2)=128;
scale_bar(:,:,2)=255;
cdata_mag(25:26,end-16:end-4,:) = scale_bar;

% % eid = 10;
% scale_bar = zeros(2,13,3,'uint8');
% % scale_bar(:,:,1)=128;
% % scale_bar(:,:,2)=128;
% scale_bar(:,:,2)=255;
% cdata_mag(25:26,end-16:end-4,:) = scale_bar;

% cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
% for kc=1:3
%     cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
% end
imwrite(cdata_mag,[str_title,'.tif']);

% %%
% colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);


%% ABO 2
% dir_video='D:\ABO\20 percent 200\';
dir_video='..\data\ABO\';
% dir_trace='..\results\ABO\unmixed traces\';
varname = '/mov';
dir_video_raw = dir_video;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% leng = 200;
% %%         
% eid = 10;
% list_nn = [16,51]; 
eid = 4;
list_nn = [55]; 
Exp_ID = list_Exp_ID{eid};
tic;
video_raw=h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
ROIs = FinalMasks;
toc;
[Lx,Ly,T] = size(video_raw);
image_max = max(video_raw,[],3);
num_neuron = size(ROIs,3);

% %%
% imshow_range = [min(image_max,[],'all'),max(image_max,[],'all')/2]; % 
imshow_range = [000,2000];
figure('Position',[480,550,480,400]);
imshow(image_max,imshow_range);  % 
hold on;

for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    [xx, yy] = meshgrid(1:Ly,1:Lx); 
    r_bg0=sqrt(mean(sum(sum(ROIs)))/pi)*2.5;
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    r_bg=max(r_bg0,sqrt(length(xxs)/pi)*2.5);
    circle = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
    half_show = 20;
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

    rect_region = [ymin_show, xmin_show, ymax_show-ymin_show+1, xmax_show-xmin_show+1];
    rectangle('Position',rect_region,'EdgeColor',colors(ind,:),'LineWidth',1);
end

for n2 = 1:num_neuron
    if ~any(n2==list_nn)
        contour(ROIs(:,:,n2),'LineColor',orange);
%         edge_mask = edge(ROIs(:,:,n2))*transparency;
%         color_all = uint8(floor(ones(Lx,Ly,3).*reshape(cyan,[1,1,3])*256));
%         imagesc(color_all, 'AlphaData', edge_mask);
    end
end
for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    contour(mask,'LineColor',blue,'LineWidth',2);
%     edge_mask = edge(mask,'nothinning')*transparency;
%     color_all = uint8(floor(ones(Lx,Ly,3).*reshape(red,[1,1,3])*256));
%     imagesc(color_all, 'AlphaData', edge_mask);
end
str_title = [Exp_ID,' max'];
title(str_title);
set(gca,'FontSize',12);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

mag=1;
% figure; 
% mag_kernel = ones(mag,mag,class(image));
% imshow(kron(image,mag_kernel),imshow_range); 
% % colorbar; 
% hold on; contour(kron(mask_show,logical(mag_kernel)),'LineColor',red);
% for n2 = 1:length(neighbors)
%     neighbor = neighbors(n2);
%     contour(kron(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),logical(mag_kernel)),'LineColor',colors(n2,:));
% end
% % contour(kron(circle(xmin_show:xmax_show,ymin_show:ymax_show),logical(mag_kernel)),'LineColor',colors(2,:))

% title(str_title);
crop_png=[86,64,size(image_max,2)*mag,size(image_max,1)*mag];
img_all=getframe(gcf,crop_png);
cdata_mag=img_all.cdata;

% eid = 4;
scale_bar = zeros(2,13,3,'uint8');
% scale_bar(:,:,1)=128;
% scale_bar(:,:,2)=128;
scale_bar(:,:,2)=255;
cdata_mag(end-5:end-4,end-16:end-4,:) = scale_bar;

% % eid = 10;
% scale_bar = zeros(2,13,3,'uint8');
% % scale_bar(:,:,1)=128;
% % scale_bar(:,:,2)=128;
% scale_bar(:,:,2)=255;
% cdata_mag(25:26,end-16:end-4,:) = scale_bar;

% cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
% for kc=1:3
%     cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
% end
imwrite(cdata_mag,[str_title,'.tif']);

% %%
% colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);


%% 1p
% dir_video='E:\OnePhoton videos\cropped videos\';
dir_video='..\data\1p\';
% dir_trace='..\results\1p\unmixed traces\';
% varname = '/mov';
% dir_video_raw = dir_video;
varname = '/network_input';
dir_video_raw = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% leng = 200;
% %%         
eid = 9;
list_nn = 1; 
Exp_ID = list_Exp_ID{eid};
tic;
video_raw=h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
ROIs = FinalMasks;
toc;
% [Lx,Ly,T] = size(video_raw);
image_max = max(video_raw,[],3);
num_neuron = size(ROIs,3);

% %%
mag=4;
mag_kernel = ones(mag,mag,class(image_max));
mag_kernel_bool = logical(mag_kernel);
image_max = kron(image_max,mag_kernel);
[Lx,Ly] = size(image_max);

% %%
% imshow_range = [min(image_max,[],'all'),max(image_max,[],'all')/2]; % 
imshow_range = [0,40];
figure('Position',[480,550,480,400]);
imshow(image_max,imshow_range);  % 
hold on;

for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=kron(ROIs(:,:,nn),mag_kernel_bool);
    [xx, yy] = meshgrid(1:Ly,1:Lx); 
    r_bg0=sqrt(mean(sum(sum(ROIs)))/pi)*2.5*mag;
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    r_bg=max(r_bg0,sqrt(length(xxs)/pi)*2.5*mag);
    circle = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
    half_show = 10*mag;
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

    rect_region = [ymin_show, xmin_show, ymax_show-ymin_show+1, xmax_show-xmin_show+1];
    rectangle('Position',rect_region,'EdgeColor',colors(ind,:),'LineWidth',1);
end

for n2 = 1:num_neuron
    if ~any(n2==list_nn)
        contour(kron(ROIs(:,:,n2),mag_kernel_bool),'LineColor',orange);
%         edge_mask = edge(kron(ROIs(:,:,n2),mag_kernel_bool))*transparency;
%         color_all = uint8(floor(ones(Lx,Ly,3).*reshape(cyan,[1,1,3])*256));
%         imagesc(color_all, 'AlphaData', edge_mask);
    end
end
for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    contour(kron(mask,mag_kernel_bool),'LineColor',blue,'LineWidth',2)
%     edge_mask = edge(kron(mask,mag_kernel_bool),'nothinning')*transparency;
%     color_all = uint8(floor(ones(Lx,Ly,3).*reshape(red,[1,1,3])*256));
%     imagesc(color_all, 'AlphaData', edge_mask);
end
str_title = [Exp_ID,' max'];
title(str_title,'Interpreter','None');
set(gca,'FontSize',12);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

% figure; 
% mag_kernel = ones(mag,mag,class(image));
% imshow(kron(image,mag_kernel),imshow_range); 
% % colorbar; 
% hold on; contour(kron(mask_show,logical(mag_kernel)),'LineColor',red);
% for n2 = 1:length(neighbors)
%     neighbor = neighbors(n2);
%     contour(kron(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),logical(mag_kernel)),'LineColor',colors(n2,:));
% end
% % contour(kron(circle(xmin_show:xmax_show,ymin_show:ymax_show),logical(mag_kernel)),'LineColor',colors(2,:))

% title(str_title);
crop_png=[86,64,size(image_max,2),size(image_max,1)];
img_all=getframe(gcf,crop_png);
cdata_mag=img_all.cdata;

scale_bar = zeros(2,12,3,'uint8');
% scale_bar(:,:,1)=128;
% scale_bar(:,:,2)=128;
scale_bar(:,:,2)=255;
cdata_mag(end-7:end-6,end-35:end-24,:) = scale_bar;

% cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
% for kc=1:3
%     cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
% end
imwrite(cdata_mag,[str_title,'.tif']);

% %%
colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','SNR');
set(h,'FontSize',14);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);


%% NAOMi
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f';
% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
dir_video='..\data\NAOMi\';
% dir_trace='..\results\NAOMi\unmixed traces\';
varname = '/mov';
dir_video_raw = dir_video;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
% leng = 200;
% %%         
eid = 7;
list_nn = 32; 
Exp_ID = list_Exp_ID{eid};
tic;
video_raw=h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
ROIs = FinalMasks;
toc;
% [Lx,Ly,T] = size(video_raw);
image_max = max(video_raw,[],3);
num_neuron = size(ROIs,3);

% %%
mag=2;
mag_kernel = ones(mag,mag,class(image_max));
mag_kernel_bool = logical(mag_kernel);
image_max = kron(image_max,mag_kernel);
[Lx,Ly] = size(image_max);

% %%
% imshow_range = [min(image_max,[],'all'),max(image_max,[],'all')/2]; % 
imshow_range = [000,1500];
figure('Position',[480,550,480,400]);
imshow(image_max,imshow_range);  % 
hold on;

for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=kron(ROIs(:,:,nn),mag_kernel_bool);
    [xx, yy] = meshgrid(1:Ly,1:Lx); 
    r_bg0=sqrt(mean(sum(sum(ROIs)))/pi)*2.5*mag;
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    r_bg=max(r_bg0,sqrt(length(xxs)/pi)*2.5*mag);
    circle = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
    half_show = 16*mag;
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

    rect_region = [ymin_show, xmin_show, ymax_show-ymin_show+1, xmax_show-xmin_show+1];
    rectangle('Position',rect_region,'EdgeColor',colors(ind,:),'LineWidth',1);
end

for n2 = 1:num_neuron
    if ~any(n2==list_nn)
        contour(kron(ROIs(:,:,n2),mag_kernel_bool),'LineColor',orange);
%         edge_mask = edge(kron(ROIs(:,:,n2),mag_kernel_bool))*transparency;
%         color_all = uint8(floor(ones(Lx,Ly,3).*reshape(cyan,[1,1,3])*256));
%         imagesc(color_all, 'AlphaData', edge_mask);
    end
end
for ind = 1:length(list_nn)
    nn=list_nn(ind);
    mask=ROIs(:,:,nn);
    contour(kron(mask,mag_kernel_bool),'LineColor',blue,'LineWidth',2)
%     edge_mask = edge(kron(mask,mag_kernel_bool),'nothinning')*transparency;
%     color_all = uint8(floor(ones(Lx,Ly,3).*reshape(red,[1,1,3])*256));
%     imagesc(color_all, 'AlphaData', edge_mask);
end
str_title = [Exp_ID,' max'];
title(str_title,'Interpreter','None');
set(gca,'FontSize',12);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);

% figure; 
% mag_kernel = ones(mag,mag,class(image));
% imshow(kron(image,mag_kernel),imshow_range); 
% % colorbar; 
% hold on; contour(kron(mask_show,logical(mag_kernel)),'LineColor',red);
% for n2 = 1:length(neighbors)
%     neighbor = neighbors(n2);
%     contour(kron(ROIs(xmin_show:xmax_show,ymin_show:ymax_show,neighbor),logical(mag_kernel)),'LineColor',colors(n2,:));
% end
% % contour(kron(circle(xmin_show:xmax_show,ymin_show:ymax_show),logical(mag_kernel)),'LineColor',colors(2,:))

% title(str_title);
crop_png=[86,64,size(image_max,2),size(image_max,1)];
img_all=getframe(gcf,crop_png);
cdata_mag=img_all.cdata;

scale_bar = zeros(2,20,3,'uint8');
% scale_bar(:,:,1)=128;
% scale_bar(:,:,2)=128;
scale_bar(:,:,2)=255;
cdata_mag(end-6:end-5,38:57,:) = scale_bar;

% cdata_mag=zeros(size(cdata,1)*mag,size(cdata,2)*mag,3,'uint8');
% for kc=1:3
%     cdata_mag(:,:,kc)=kron(cdata(:,:,kc),mag_kernel);
% end
imwrite(cdata_mag,[str_title,'.tif']);

% %%
colormap gray; axis('image'); 
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
% saveas(gcf,[str_title,'.png']);
% saveas(gcf,[str_title,'.emf']);
