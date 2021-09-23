function [bgtraces,traces]=generate_bgtraces_from_masks(video,masks)
% Generate background traces for each neuron from ground truth masks
[Lx,Ly,T]=size(video);
[Lxm,Lym,ncells]=size(masks);

[xx, yy] = meshgrid(1:Ly,1:Lx); 
r_bg=sqrt(mean(sum(sum(masks)))/pi)*2.5;

if Lx==Lxm && Ly==Lym
    video=reshape(video,[Lxm*Lym,T]);
else
    video=reshape(video(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:),[Lxm*Lym,T]);
end

traces=zeros(ncells,T); %,'single'
bgtraces=zeros(ncells,T); %,'single'
for nn=1:ncells
    mask = masks(:,:,nn);
    [xxs,yys]=find(mask>0);
    comx=mean(xxs);
    comy=mean(yys);
    circleout = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
    bgtraces(nn,:)=median(video(circleout(:),:),1);
    traces(nn,:)=mean(video(mask(:),:),1);
end


