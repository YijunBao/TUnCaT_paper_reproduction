alpha = 0.02; % [0.006, .008, .01:.01:.1];
nindneurons = 0;
trace=dff';
[ncells,T]=size(trace);
neighbors=1:ncells;
bgtrace=zeros(1,T);
for j = 1:length(alpha)
    [temptraceout mixout] = nmfunmix_v6(cat(1,trace(neighbors,:),bgtrace),alpha(j),0.5);
    temp_nindneurons = sum(double(sum(mixout,2) > 0));
    if temp_nindneurons < nindneurons || (temp_nindneurons == 1 && j > 1)
        fprintf('%f\n',alpha(j-1));
        [traceout mixout] = nmfunmix_v6(cat(1,trace(neighbors,:),bgtrace),alpha(j-1),0.5);
        break;
    else
        nindneurons = temp_nindneurons;
        traceout = temptraceout;
    end
    %pause
end

%%
dir_video='D:\ABO\20 percent 200\';
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
ii = 10;
Exp_ID = list_Exp_ID{ii};
alpha = 0.3;

load([dir_video,'traces_mix_SNR_FISSAinputunmix_sigma1_\raw\',Exp_ID,'.mat'],'traces','bgtraces');
load([dir_video,'traces_FISSA_SNR (tol=1e-4, max_iter=20000)\raw\',Exp_ID,'.mat'],'raw_traces');
raw_traces = raw_traces';
mse(traces-raw_traces)

load(sprintf('%straces_mix_SNR_FISSAinputunmix_sigma1_\\alpha=%6.3f\\%s.mat',dir_video,alpha,Exp_ID),'traces_nmfdemix');
load(sprintf('%straces_FISSA_SNR (tol=1e-4, max_iter=20000)\\alpha=%5.2f\\%s.mat',dir_video,alpha,Exp_ID),'unmixed_traces');
unmixed_traces = unmixed_traces';
mse(unmixed_traces-traces_nmfdemix)

%%
figure; 
plot(trace); 
hold on; 
plot(spikes_frames_now(~GT_match),-1*ones(sum(~GT_match),1),'+r'); 
plot(spikes_frames_now(GT_match),-1*ones(sum(GT_match),1),'+g'); 
plot(locs, peaks);
spikes_eval_now_4=[spikes_eval_now,eval_match];

%%
load('C:\Matlab Files\Unmixing\simulation\scores_CNMF_100s_30Hz_100+10_burst3_RawVideo_p1_pureSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_CNMF_100s_30Hz_100+10_burst3_RawVideo_p1_sumSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_CNMF_100s_30Hz_100+10_burst3_SNRVideo_p1_pureSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_CNMF_100s_30Hz_100+10_burst3_SNRVideo_p1_sumSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))

load('C:\Matlab Files\Unmixing\simulation\scores_ours_100s_30Hz_100+10_burst3_RawVideo_Raw_compSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_ours_100s_30Hz_100+10_burst3_RawVideo_Unmix_compSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_ours_100s_30Hz_100+10_burst3_SNRVideo_Raw_compSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))
load('C:\Matlab Files\Unmixing\simulation\scores_ours_100s_30Hz_100+10_burst3_SNRVideo_Unmix_compSigma.mat')
mean_F1 = squeeze(mean(list_F1,1)); disp(max(mean_F1(:)))

%%
folder = 'F:\NAOMi\100s_30Hz_N=400_40mW_noise10+23\GT Masks\';
list_ncells = zeros(1,10);
for vid = 0:9
    file = [folder,sprintf('FinalMasks_Video_%d_sparse.mat',vid)];
    load(file,'GTMasks_2');
    list_ncells(vid+1) = size(GTMasks_2,2);
end
disp(mean(list_ncells))

%%
folder = 'F:\NAOMi\100s_30Hz_N=400_40mW_noise10+23\';
vid = 9;
video_raw = h5read([folder,'Video_',num2str(vid),'.h5'],'/mov');
video_SNR = h5read([folder,'SNR Video\Video_',num2str(vid),'.h5'],'/network_input');
load([folder,'GT Masks\FinalMasks_Video_',num2str(vid),'.mat'],'FinalMasks');
load([folder,'GT Masks\Traces_etc_Video_',num2str(vid),'.mat'],'Masks','clean_traces','spikes');
%%
frame_raw = 1156;
frame_SNR = 1155;
xrange = 1:40;
yrange = 1:50;
nn = 42;

figure('Position',[600,100,500,400]); 
imagesc(video_raw(xrange,yrange,frame_raw),[0,max(max(max(video_raw)))/20]);
axis('image'); colorbar; colormap gray;
hold on;
contour(FinalMasks(xrange,yrange,nn),'r');
title('Raw frame','FontSize',12);
saveas(gcf,sprintf('Raw frame %d Neuron %d Video %d.png',frame_raw,nn,vid));

figure('Position',[100,100,500,400]); 
imagesc(video_SNR(xrange,yrange,frame_SNR),[0,max(max(max(video_SNR)))/5]);
axis('image'); colorbar; colormap gray;
hold on;
contour(FinalMasks(xrange,yrange,nn),'r');
title('SNR frame','FontSize',12);
saveas(gcf,sprintf('SNR frame %d Neuron %d Video %d.png',frame_SNR,nn,vid));

%%
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
num_Exp = length(list_Exp_ID);
alpha = 3;
addon = '_pertmin=0.16_eps=0.1_direction'; % '_eps=0.1'; % 
dir_video='D:\ABO\20 percent 200';
method = 'ours'; % {'FISSA','ours'}
video='SNR'; % {'Raw','SNR'}
folder = sprintf('traces_%s_%s%s',method,video,addon);
dir_FISSA = fullfile(dir_video,folder);
list_final_alpha_all = cell(1,num_Exp);
for ii = 1:num_Exp
    Exp_ID = list_Exp_ID{ii};
    load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'list_final_alpha');
    list_final_alpha_all{ii} = double(list_final_alpha);
end
list_final_alpha_all_mat = cell2mat(list_final_alpha_all);

figure; 
division_time = log2(alpha./list_final_alpha_all_mat);
histogram(division_time,0:1:max(division_time));
xlabel('Division time');
set(gca,'FontSize',14);
saveas(gcf,sprintf('ABO division time %s %6.3f.png',video,alpha));

%%
% simu_opt = '1100s_3Hz_N=200_40mW_noise10+23_NA0.4,0.3_jGCaMP7c'; % _NA0.4,0.3
% dir_video=['F:\NAOMi\',simu_opt,'\']; % 
% list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
% addon = '_pertmin=0.16_eps=0.1_range_FinalAlpha'; % '_eps=0.1'; % 

dir_video='E:\OnePhoton videos\cropped videos\';
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
addon = '_range2_merge'; % '_eps=0.1'; % 

method = 'ours'; % {'FISSA','ours'}
video='SNR'; % {'Raw','SNR'}
folder = sprintf('traces_%s_%s%s',method,video,addon);
dir_FISSA = fullfile(dir_video,folder);
num_Exp = length(list_Exp_ID);
alpha = 5;
list_final_alpha_all = cell(1,num_Exp);
for ii = 1:num_Exp
    Exp_ID = list_Exp_ID{ii};
    load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'list_final_alpha');
    list_final_alpha_all{ii} = double(list_final_alpha);
end
list_final_alpha_all_mat = cell2mat(list_final_alpha_all);

figure; 
division_time = log2(alpha./list_final_alpha_all_mat);
histogram(division_time,1:max(division_time));
xlabel('Division time');
set(gca,'FontSize',14);
list_question = cellfun(@(x) find(x==min(list_final_alpha_all_mat)), list_final_alpha_all, 'UniformOutput', false);
list_question_more = cellfun(@(x) find(x<min(list_final_alpha_all_mat)*2^8), list_final_alpha_all, 'UniformOutput', false);
% temp = list_traces_unmixed_filt{3,1}(8,:);
% min(temp)+max(temp) - 2*median(temp)
% mean(temp) - median(temp)
title(sprintf('%s video, totally %d neurons',video,length(division_time)),'Fontsize',14);
% saveas(gcf,sprintf('division time 1p %s %6.3f.png',video,alpha));

%%
load('F:\NAOMi_hasStart\1000s_3Hz_N=200_40mW_noise10+23_NA0.4,0.3_jGCaMP7c\traces_ours_Raw_fixed_alpha\raw\Video_9.mat')
nn=44; 
trace = traces(:,nn)-bgtraces(:,nn);
T = length(trace);
figure; 
plot(trace);
hold on;
plot(1:T,median(trace)*ones(1,T));
[f,xi] = ksdensity(trace);
[~,pos] = max(f); 
plot(1:T,xi(pos)*ones(1,T));
xlabel('Time (frame)')
legend('Trace','Median','KSD peak');
set(gca,'FontSize',14);
saveas(gcf,'Example KSD baseline.png')

%%
video = 'SNR'; % 'Raw'; % 
list_nbin = 2.^(1:7);
Lb = length(list_nbin);
Table_time_mean = cell(Lb,1);
for ib = 1:Lb
    nbin = list_nbin(ib);
    if nbin==1
        load(['D:\ABO\20 percent 200\traces_ours_',video,'\Table_time.mat'])
    else
        load(['D:\ABO\20 percent 200\traces_ours_',video,'_BinUnmix_downsample',num2str(nbin),'\Table_time.mat'])
    end
    Table_time_mean{ib} = mean(Table_time(:,1:end-1),1);
end
Table_time_mean = cell2mat(Table_time_mean);
figure; 
plot(list_alpha,Table_time_mean,'LineWidth',2);
xlabel('alpha')
ylabel('Time (s)')
title([video,' videos']);
set(gca,'XScale','log');
legend(arrayfun(@(x) ['nbin=',num2str(x)], list_nbin, 'UniformOutput',false));
set(gca,'FontSize',14);
saveas(gcf,['time alpha nbin ABO ',video,'.png']);

%% plot time vs alpha for nbin 
video = 'Raw'; % 'SNR'; % 
load(['D:\ABO\20 percent 200\traces_ours_',video,'_downsample128_numba\Table_time.mat'],'Table_time')
Table_time_numba = Table_time(:,1:end-1)+Table_time(:,end); 
load(['D:\ABO\20 percent 200\traces_ours_',video,'_downsample128_shm\Table_time.mat'],'Table_time')
Table_time_shm = Table_time(:,1:end-1)+Table_time(:,end); 
load(['D:\ABO\20 percent 200\traces_ours_',video,'_BinUnmix_downsample128\Table_time.mat'],'Table_time')
Table_time_old = Table_time(:,1:end-1)+Table_time(:,end); 
Table_time_mean_all = [mean(Table_time_old,1); mean(Table_time_shm,1); mean(Table_time_numba,1)];

figure; 
plot(list_alpha', Table_time_mean_all','LineWidth',2);
legend({'outside trace in unmix','outside trace before unmix','outside trace before unmix v2'});
xlabel('alpha')
ylabel('Time (s)')
title([video,' videos']);
set(gca,'XScale','log');
legend({'outside trace in unmix','outside trace before unmix','outside trace before unmix (numba)'},'Location','East');
set(gca,'FontSize',14);
saveas(gcf,['time alpha out_or_in ABO ',video,'.png']);

%% plot time vs alpha for nbin 
figure; 
video = 'SNR'; % 'Raw'; % 
dir_video = 'D:\ABO\20 percent 200\';
% dir_video = 'E:\OnePhoton videos\cropped videos\';
% dir_video = 'F:\NAOMi\110s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6s\';
% dir_video = 'F:\NAOMi\1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c\';
load([dir_video,'traces_ours_',video,'\Table_time.mat'],'Table_time','list_alpha')
Table_time_in = Table_time(:,1:end-1)+Table_time(:,end); 
plot(list_alpha, mean(Table_time_in,1)','LineWidth',2);
hold on;
load([dir_video,'traces_ours_',video,'_novideounmix_r2\Table_time.mat'],'Table_time','list_alpha')
Table_time_before = Table_time(:,1:end-1)+Table_time(:,end); 
plot(list_alpha, mean(Table_time_before,1)','LineWidth',2);
% Table_time_mean_all = [mean(Table_time_in,1); mean(Table_time_before,1)];

% legend({'outside trace in unmix','outside trace before unmix'});
xlabel('alpha')
ylabel('Time (s)')
title([video,' videos']);
set(gca,'XScale','log');
legend({'outside trace in unmix','outside trace before unmix'}); % ,'Location','East'
set(gca,'FontSize',14);
% saveas(gcf,['time alpha out_or_in jGCaMP7c ',video,' 0616.png']);

%%
addpath(genpath('C:\Matlab Files\Filter'));
dir_video='D:\ABO\20 percent 200\';
varname = '/mov';
dir_video_SNR = dir_video;
% varname = '/network_input';
% dir_video_SNR = fullfile(dir_video, 'SNR video');
dir_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
eid = 1;
Exp_ID = list_Exp_ID{eid};
video=h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),varname);
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');

[traces0,bgtraces_mean,bgtraces_median]=generate_bgtraces_from_masks_3(video,FinalMasks);

%%
new = load('demixtest_new.mat');
old = load('demixtest_old.mat');
% traces_target_new = cell2mat(cellfun(@(x) x(1,:), new.list_tracein, 'UniformOutput', false)');
bgtraces_out_new = cell2mat(cellfun(@(x) x(end,:), new.list_tracein, 'UniformOutput', false)');
bgtraces_out_old = cell2mat(cellfun(@(x) x(end,:), old.list_tracein, 'UniformOutput', false)');
max_diff = cellfun(@(x,y) max(max(abs(x-y))), new.list_tracein,old.list_tracein)';

% figure; imagesc(bgtraces_mean-bgtraces_median-bgtraces_out_old); colorbar;
figure; imagesc(bgtraces_out_new-bgtraces_out_old); colorbar;
% figure; imagesc(traces0-bgtraces_median-traces_target_new); colorbar;

%%
nn=15; 
figure; 
subplot(2,1,1); 
plot(raw_new.traces_nmfdemix(:,nn)); 
subplot(2,1,2); 
plot(raw_old.traces_nmfdemix(:,nn));
%%
nn=39; 
figure; 
subplot(2,1,1); 
plot(bgtraces_out_new(nn,:)-bgtraces_out_old(nn,:)); 
subplot(2,1,2); 
plot(bgtraces_median(nn,:));
%%
nn=15; 
figure; 
subplot(3,1,1); 
plot(raw_old.traces_nmfdemix(:,nn)); 
subplot(3,1,2); 
plot(raw_new.traces_nmfdemix(:,nn)); 
subplot(3,1,3); 
plot(raw_old.traces_nmfdemix(:,nn)-raw_new.traces_nmfdemix(:,nn)); 

%%
raw_old=load('E:\OnePhoton videos\cropped videos\traces_ours_Raw_merge\raw\c25_59_228.mat');
raw_new=load('E:\OnePhoton videos\cropped videos\traces_ours_Raw_merge_novideounmix\raw\c25_59_228.mat');
max(max(raw_new.bgtraces-raw_old.bgtraces))
figure; bar(max((raw_new.bgtraces-raw_old.bgtraces),[],1)')
%%
raw_old=load('E:\OnePhoton videos\cropped videos\traces_ours_Raw_merge\alpha= 1.000\c25_59_228.mat');
raw_new=load('E:\OnePhoton videos\cropped videos\traces_ours_Raw_merge_novideounmix\alpha= 1.000\c25_59_228.mat');
max(max(raw_new.traces_nmfdemix-raw_old.traces_nmfdemix))
figure; bar(max((raw_new.traces_nmfdemix-raw_old.traces_nmfdemix),[],1)')


%% count the number of neurons in ABO
dir_video='D:\ABO\20 percent\';
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
num_Exp = length(list_Exp_ID);
list_ncells = zeros(num_Exp,1);
for eid = 1:num_Exp
    load([dir_video,'traces_ours_Raw_novideounmix_r2_mixout\raw\',list_Exp_ID{eid},'.mat'],'traces');
    list_ncells(eid) = size(traces,2);
%     load(['manual labels\output_',list_Exp_ID{eid},'.mat'],'output');
%     list_ncells(eid) = length(output);
end
[mean(list_ncells), std(list_ncells,1)]

%% Save processing time with different alpha 
dir_video='D:\ABO\20 percent 200\';
list_video = {'Raw','SNR'};
list_addon = {'_novideounmix_r2_mixout1000','_novideounmix_r2_fixed_alpha'};
num_video = length(list_video);
num_addon = length(list_addon);
[list_alpha_all, Table_time_all] = deal(cell(num_addon, num_video));
load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
Table_time_SNR = Table_time';
for vid = 1:num_video
    video = list_video{vid};
    for aid = 1:num_addon
        addon = list_addon{aid};
        load([dir_video,'traces_ours_',video,addon,'\Table_time.mat'],'list_alpha','Table_time')
        if contains(video,'SNR')
            Table_time(:,end) = Table_time(:,end) + Table_time_SNR;
        end
        list_alpha_all{aid,vid} = list_alpha;
        Table_time_all{aid,vid} = Table_time;
    end
end
save('Time_alpha_ABO.mat','list_alpha_all','Table_time_all','list_video','list_addon');


%% Save processing time with different alpha 
dir_video='D:\ABO\20 percent 200\';
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
num_Exp=length(list_Exp_ID);
list_video = {'Raw','SNR'};
list_addon = {'_novideounmix_r2_mixout1000','_novideounmix_r2_fixed_alpha'};
num_video = length(list_video);
num_addon = length(list_addon);
[list_alpha_all, list_n_iter_all] = deal(cell(num_addon, num_video));
for vid = 1:num_video
    video = list_video{vid};
    for addid = 1:num_addon
        addon = list_addon{addid};
        dir_path = [dir_video,'traces_ours_',video,addon];
        load([dir_path,'\Table_time.mat'],'list_alpha')
        list_alpha_all{addid,vid} = list_alpha;
        num_alpha = length(list_alpha);
        Table_n_iter = zeros(num_Exp, num_alpha);
        for eid = 1:num_Exp
            Exp_ID = list_Exp_ID{eid};
            for aid = 1:num_alpha
                alpha = list_alpha(aid);
                load(sprintf('%s\\alpha=%6.3f\\%s.mat',dir_path,alpha,Exp_ID),'list_n_iter');
                Table_n_iter(eid,aid) = mean(list_n_iter);
            end
        end
        list_n_iter_all{addid,vid} = Table_n_iter;
    end
end
save('Niter_alpha_ABO.mat','list_alpha_all','list_n_iter_all','list_video','list_addon');

%%
noresidual = load('C:\Matlab Files\Unmixing\include\scores_split_ours_RawVideo_noresidual_fixed_alpha_UnmixSigma_ksd-psd.mat');
residual = load('C:\Matlab Files\Unmixing\include\scores_split_ours_RawVideo_novideounmix_r2_fixed_alpha_UnmixSigma_ksd-psd.mat');
max(abs(residual.list_F1(:,1:16,:)-noresidual.list_F1),[],'all')
diff = residual.list_F1(:,1:16,:)-noresidual.list_F1;
diff_not0 = diff(diff~=0);
figure; histogram(diff_not0)
figure; imagesc(squeeze(mean(diff,1))); colorbar;
%%
noresidual = load('C:\Matlab Files\Unmixing\NAOMi\scores_split_ours_120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f_SNRVideo_Unmix_compSigma_noresidual_fixed_alpha_ksd-psd.mat');
residual = load('C:\Matlab Files\Unmixing\NAOMi\scores_split_ours_120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f_SNRVideo_Unmix_compSigma_novideounmix_r2_fixed_alpha_ksd-psd.mat');
max(abs(residual.list_F1-noresidual.list_F1),[],'all')
diff = residual.list_F1-noresidual.list_F1;
figure; histogram(diff(:))
figure; imagesc(squeeze(mean(diff,1))); colorbar;
%%
noresidual = load('C:\Matlab Files\Unmixing\1p\scores_split_ours_SNRVideo_merge_noresidual_fixed_alpha_UnmixSigma_ksd-psd.mat');
residual = load('C:\Matlab Files\Unmixing\1p\scores_split_ours_SNRVideo_merge_novideounmix_r2_fixed_alpha_UnmixSigma_ksd-psd.mat');
max(abs(residual.list_F1-noresidual.list_F1),[],'all')
diff = residual.list_F1-noresidual.list_F1;
figure; histogram(diff(:))
figure; imagesc(squeeze(mean(diff,1))); colorbar;


%% mean areas of neurons in 1p
dir_video='E:\OnePhoton videos\cropped videos\';
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
num_Exp = length(list_Exp_ID);
[list_ncells] = deal(zeros(num_Exp,1));
[list_areas] = deal(cell(num_Exp,1));
for eid = 1:num_Exp
    load([dir_video,'GT Masks merge\FinalMasks_',list_Exp_ID{eid},'.mat'],'FinalMasks');
    list_ncells(eid) = size(FinalMasks,3);
    list_areas{eid} = squeeze(sum(sum(FinalMasks,1),2));
end
areas_all = cell2mat(list_areas);
[mean(areas_all), std(areas_all,1)]

%% Distribution of number of attempted alpha
supertitle = 'ABO'; % {'ABO','NAOMi','One-photon'};
% list_addon = {'_th_pertmin=1_0mixout_overlap','_th_pertmin=1_0mixout',...
%     '_pertmin=1_coact=3','_pertmin=0.25_coact=3'}; %,...
% list_legend = {['no 0 in mixout',newline,'for overlapped'],'no 0 in mixout',...
%     'no coactive',['no coactive,',newline,'pertmin<0.25']}; %,...
list_addon = {'_pertmin=1_residual=0.01_2side','_pertmin=1_residual0=0.01_2side','_pertmin=0.05_residual=0_2side'}; %'_fixed_alpha',
list_legend = {'residual=0.01','residual0=0.01','pertmin=0.05'}; %'fixed alpha',
dir_video='D:\ABO\20 percent 200\';
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};

% supertitle = 'NAOMi'; % {'ABO','NAOMi','One-photon'};
% % list_addon = {'_th_pertmin=1_0mixout_overlap','_th_pertmin=1_0mixout','_th_pertmin=1_noise0.01','_th_pertmin=0.05'}; %,'_fixed_alpha'
% % list_legend = {['no 0 in mixout',newline,'for overlapped'],'no 0 in mixout','residual<=0.01','pertmin<=0.05'}; %,'fixed alpha'
% list_addon = {'_pertmin=1_residual=0.01_2side','_pertmin=0.05_residual=0_2side','_th_pertmin=1_noise0.01','_th_pertmin=0.05'}; %,'_fixed_alpha'
% list_legend = {'residual=0.01','pertmin=0.05','residual<=0.01','pertmin<=0.05'}; %,'fixed alpha'
% simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % 
% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
% list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);

% supertitle = 'One-photon'; % {'ABO','NAOMi','One-photon'};
% % list_addon = {'_th_pertmin=1_0mixout_overlap','_th_pertmin=1_0mixout','_th_pertmin=1_noise0.01','_th_pertmin=0.05'}; %,'_fixed_alpha'
% % list_legend = {['no 0 in mixout',newline,'for overlapped'],'no 0 in mixout','residual<=0.01','pertmin<=0.05'}; %,'fixed alpha'
% list_addon = {'_pertmin=1_residual=0.01_2side','_pertmin=0.05_residual=0_2side','_th_pertmin=1_noise0.01','_th_pertmin=0.05'}; %,'_fixed_alpha'
% list_legend = {'residual=0.01','pertmin=0.05','residual<=0.01','pertmin<=0.05'}; %,'fixed alpha'
% dir_video='E:\OnePhoton videos\cropped videos\';
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};

alpha = 1;
list_video = {'Raw','SNR'};
num_Exp = length(list_Exp_ID);
num_video = length(list_video);
num_addon = length(list_addon);
figure('Position',[0,50,1420,850]); 
gap = [0.12,0.1];  marg_h = 0.08; marg_w = 0.08; 
ha = tight_subplot(num_video, num_addon, gap, marg_h, marg_w);
clear ax;

for aid = 1:num_addon
    addon = list_addon{aid};
    for vid = 1:num_video
        video = list_video{vid};
        dir_folder = sprintf('%straces_ours_%s%s\\alpha=%6.3f\\',dir_video,video,addon,alpha);
        list_divides = cell(1,num_Exp);
        for eid = 1:num_Exp
            load([dir_folder,list_Exp_ID{eid},'.mat'],'list_final_alpha');
            list_final_alpha(list_final_alpha==0)=alpha;
            list_divides{eid} = round(log2(alpha./double(list_final_alpha)));
        end
        divides_all = cell2mat(list_divides);
        [mean(divides_all), std(divides_all,1)];
        axes(ha(sub2ind([num_addon, num_video], aid,vid)));
        ax(vid,aid) = gca;
%         subplot(num_video,num_addon,(vid-1)*num_addon+aid); 
        histogram(divides_all); % ,0:20
        xlabel('Number of divisions');
        ylabel('Counts');
        title([list_legend{aid},', ',video,' videos']);
        set(gca,'FontSize',12);
    end
end

linkaxes(ax,'xy');
suptitle(supertitle);
% saveas(gcf,[supertitle,', Final alpha divisions bad methods.png'])
saveas(gcf,[supertitle,', Final alpha divisions good methods.png'])

%% variation among trials
list_trial = 0:9;
list_F1_all = zeros(10,10,11);
for t = 1:10
    score_name = ['.\include\scores_split_ours_RawVideo_alpha=1_',num2str(list_trial(t)),'_UnmixSigma_ksd-psd.mat'];
    load(score_name,'list_F1');
    list_F1_all(:,t,:) = list_F1;
end
list_F1_std = std(list_F1_all,1,2);
squeeze(list_F1_std);

%% variation among trials
list_trial = 0:9;
list_unmixed_traces = zeros(10,10,11);
for t = 1:10
    score_name = ['.\include\scores_split_ours_RawVideo_alpha=1_',num2str(list_trial(t)),'_UnmixSigma_ksd-psd.mat'];
    load(score_name,'list_F1');
    list_F1_all(:,t,:) = list_F1;
end
list_F1_std = std(list_F1_all,1,2);
squeeze(list_F1_std);

%% Number of divisions for optimized alpha
% supertitle = 'ABO'; % {'ABO','NAOMi','One-photon'};
% spike_type = 'include';
% dir_video='D:\ABO\20 percent 200\';
% addon = '_novideounmix_r2_mixout1000'; % '_novideounmix_r2_mixout'; % 
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};

% supertitle = 'NAOMi'; % {'ABO','NAOMi','One-photon'};
% spike_type = 'NAOMi';
% simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % 
% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
% addon = '_novideounmix_r2_mixout'; % 
% list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);

supertitle = 'One-photon'; % {'ABO','NAOMi','One-photon'};
spike_type = '1p';
dir_video='E:\OnePhoton videos\cropped videos\';
addon = '_merge_novideounmix_r2_mixout'; % 
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};

% alpha = 1;
list_video = {'Raw','SNR'};
num_Exp = length(list_Exp_ID);
num_video = length(list_video);
% num_addon = length(list_addon);
figure('Position',[0,50,800,450]); 
% gap = [0.12,0.1];  marg_h = 0.08; marg_w = 0.08; 
% ha = tight_subplot(num_video, num_addon, gap, marg_h, marg_w);
% clear ax;

for vid = 1:num_video
    video = list_video{vid};
    list_divides = cell(1,num_Exp);
    if contains(spike_type,'NAOMi')
        scorefile = ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_compSigma',addon,'_ksd-psd','.mat'];
    else
        scorefile = ['scores_split_ours_',video,'Video',addon,'_UnmixSigma','_ksd-psd','.mat'];
    end
    load(fullfile('.\',spike_type,scorefile),'list_F1','list_alpha','list_thred_ratio');
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    alpha_CV = zeros(1,num_Exp);
    
    for eid = 1:num_Exp
        CV = eid;
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
%         F1_CV(CV,mid,vid) = list_F1_2(CV,ind_param);
        [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
        alpha = list_alpha(ind_alpha);
        alpha = 1;
        alpha_CV(CV) = alpha;
%         thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_thred_ratio);
%         Table_F1_CV{mid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
%         ind_alpha = find(list_alpha_time==alpha);
        
        dir_folder = sprintf('%straces_ours_%s%s\\alpha=%6.3f\\',dir_video,video,addon,alpha);
        load([dir_folder,list_Exp_ID{eid},'.mat'],'list_final_alpha');
        list_final_alpha(list_final_alpha==0)=alpha;
        list_divides{eid} = round(log2(alpha./double(list_final_alpha)));
    end
    divides_all = cell2mat(list_divides);
    [mean(divides_all), std(divides_all,1)];
%         axes(ha(sub2ind([num_addon, num_video], aid,vid)));
%         ax(vid,aid) = gca;
    ax(vid) = subplot(1,num_video,vid); 
    histogram(divides_all); % ,0:20
    [length(divides_all),sum(divides_all)]
    xlabel('Number of divisions');
    ylabel('Counts');
%     title([list_legend{aid},', ',video,' videos']);
    set(gca,'FontSize',12);
end

linkaxes(ax,'xy');
% suptitle(supertitle);
saveas(gcf,[supertitle,', Final alpha divisions 1.png'])
