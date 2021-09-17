clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
list_prot = {'GCaMP6f'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
% list_prot = {'GCaMP6s','jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
for pid = 1:length(list_prot) % [2,4,8,16,32,64] % 
    prot = list_prot{pid};
% for fs = 30 % [3, 10, 100, 300] % 
% for T = [30, 50, 320, 1020] % 1100 % 
% for N = [50, 100, 300, 400] % 
% for power = [10, 20, 30, 50, 70, 150] % [1, 3, 100] % 
for noise = [0, 0.1, 0.3] % [3, 10, 30, 50, 100] % 
%     if number == 30 && nbin == 16
%         continue;
if contains(prot,'6')
    fs = 30; % [90,300] % 3,10,
%     simu_opt = sprintf('120s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_%s',fs,prot); % 
%     simu_opt = sprintf('%ds_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_%s',T,prot); % 
%     simu_opt = sprintf('120s_30Hz_N=%d_100mW_noise10+23_NA0.8,0.6_%s',N,prot); % 
%     simu_opt = sprintf('120s_30Hz_N=200_%dmW_noise10+23_NA0.8,0.6_%s',power,prot); % 
    simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23x%s_NA0.8,0.6_%s',num2str(noise),prot); % 
else
    fs = 3;
    simu_opt = sprintf('1100s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_%s',fs,prot); %
end
% simu_opt = '100s_30Hz_N=400_40mW_noise10+23'; % 
simu_opt_split = split(simu_opt,'_');

dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
num_Exp=length(list_Exp_ID);

list_spike_type = {'NAOMi'}; % {'only','include','exclude'};
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'}; % 
list_baseline_std = {'_ksd-psd'}; % '', 
% std_method = 'quantile-based std comp';  % comp

method = 'AllenSDK'; % {'FISSA','ours','CNMF','AllenSDK'}
list_video={'Raw','SNR'}; % 
addon = '';

% list_thred_ratio=6:0.5:9; % 6:12; % 9:16; % 
% num_ratio=length(list_thred_ratio);

load(['filter_template 100Hz ',prot,'_ind_con=10.mat'],'template'); % _ind_con=10
fs_template = 100;
% load('filter_template 30Hz jGCaMP7s.mat','template');
% fs_template = 30;
[val_max, loc_max] = max(template);
peak_time = loc_max/fs_template;
loc_e = find(template>max(template)*exp(-1),1,'last');
decay = (loc_e - loc_max)/fs_template;
loc_21 = find(template>max(template)/2,1,'first');
loc_22 = find(template>max(template)/2,1,'last');
lag0 = [loc_21, loc_22]*fs/fs_template;

% load('filter_template 100Hz.mat','template');
% fs_template = 100;
% rise = 0.07;
% decay = 2.07;
% if rise>0
%     peak_time = rise*log(decay/rise+1) * fs;
% else
%     peak_time=0;
% end
% lag0 = 1+[peak_time/2, fs*decay/2];

[~,peak] = max(template);
peak = peak - 1;
leng = length(template);
xp = ((-peak):(leng-1-peak))/fs_template;
x = (round((-peak)*fs/fs_template) : round((leng-peak)*fs/fs_template))/fs;
Poisson_filt = interp1(xp,template,x,'linear','extrap');
Poisson_filt = Poisson_filt(Poisson_filt>=(max(Poisson_filt)*exp(-1)));
Poisson_filt = Poisson_filt/sum(Poisson_filt);
kernel=fliplr(Poisson_filt);

% lag = ceil(fs*rise) + ceil(fs*rise*3); % 3;
cons = ceil(fs*decay*0.1);
load([dir_video,'\GT Masks\Traces_etc_',list_Exp_ID{1},'.mat'],'clean_traces')
folder = sprintf('traces_%s_%s%s%s%s%s',method,'SNR',addon);
dir_FISSA = fullfile(dir_video,folder);
load(fullfile(dir_FISSA,[list_Exp_ID{1},'.mat']),'roi_traces','compensated_traces'); % ,'unmixed_traces'
length_kernel_py = size(clean_traces,2) - size(roi_traces,2)+1;
length_diff = length_kernel_py - length(kernel);
if length_diff > 0
    kernel = padarray(kernel,[0,length_diff],'replicate','pre');
elseif length_diff < 0
    kernel = kernel(1-length_diff:end);
end
            

%%
for bsid = 1:length(list_baseline_std)
    baseline_std = list_baseline_std{bsid};
if contains(baseline_std,'psd')
    std_method = 'psd';  % comp
    if contains(baseline_std,'ksd')
        baseline_method = 'ksd';
    else
        baseline_method = 'median';
    end
else
    std_method = 'quantile-based std comp';
    baseline_method = 'median';
end

for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110; % GCaMP6f;
                    else
                        list_thred_ratio=10:10:110;% jGCaMP7
                    end
                else
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110;% 0:10; % GCaMP6f;
%                         list_thred_ratio=60:10:160;% 0:10; % GCaMP6s;
                    else
                        list_thred_ratio=10:10:110;% 0:0.5:5; % jGCaMP7
                    end
                end
            else
                if contains(prot,'6')
                    list_thred_ratio=1:0.5:6; % 6:16; % 
                else
                    list_thred_ratio=4:14; % 6:16; %
                end
            end
            if contains(prot,'6')
                list_thred_ratio = list_thred_ratio*sqrt(fs/30);
            end
            folder = sprintf('traces_%s_%s%s%s%s%s',method,video,addon);
            dir_FISSA = fullfile(dir_video,folder);
            useTF = contains(video, 'Raw');
            num_ratio=length(list_thred_ratio);

            [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));
            [list_corr_unmix,list_MSE_all,list_MSE_rmmean,list_MSE_rmmedian, ...
                list_pct_min,list_corr_unmix_active,list_corr_unmix_inactive]...
                =deal(cell(num_Exp, 1));

        %%
        fprintf('NeuroTool');
        for ii = 1:num_Exp
            Exp_ID = list_Exp_ID{ii};
            fprintf('\b\b\b\b\b\b\b\b\b%s: ',Exp_ID);
            load([dir_video,'\GT Masks\FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
            load([dir_video,'\GT Masks\Traces_etc_',Exp_ID,'.mat'],'spikes','clean_traces')
            ncells = size(FinalMasks,3);
%                 spikes = spikes(1:ncells)';
            calcium = clean_traces;
            spikes_cell = mat2cell(spikes,ones(ncells,1),size(spikes,2));
            spikes_frames = cellfun(@(x) find(x)*fs/100, spikes_cell, 'UniformOutput', false);
%                 ncells = length(calcium);
            if useTF
                calcium_filt = conv2(calcium,kernel,'valid');
            else
                calcium = conv2(calcium,kernel,'valid');
                calcium_filt = calcium;
            end
%                 calcium = cell2mat(calcium);
            [output, spikes_GT_line] = GT_transient_NAOMi_split(calcium_filt,spikes_frames);

            load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'roi_traces','compensated_traces'); % ,'unmixed_traces'
%             traces_raw = roi_traces;
            traces_unmixed = compensated_traces;
            if useTF
                traces_unmixed_filt=conv2(traces_unmixed,kernel,'valid');
            else
                traces_unmixed_filt=traces_unmixed;
            end
            [mu, sigma] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);
            [corr_unmix,corr_unmix_active,corr_unmix_inactive] = deal(zeros(ncells,1));
                clip_pct = prctile(traces_unmixed,84,2);
%             clip_pct = median(traces_unmixed,2);
            for n = 1:ncells
%                     corr_unmix(n) = corr(max(clip_pct(n),traces_unmixed(n,:)'),calcium(n,:)');
%                     corr_unmix(n) = corr(max(mu(n),traces_unmixed(n,:)'),calcium(n,:)');
                corr_unmix(n) = corr(traces_unmixed(n,:)',calcium(n,:)');
                corr_unmix_active(n) = corr(traces_unmixed(n,spikes_GT_line(n,:))',calcium(n,spikes_GT_line(n,:))');
                corr_unmix_inactive(n) = corr(traces_unmixed(n,~spikes_GT_line(n,:))',calcium(n,~spikes_GT_line(n,:))');
            end
            list_corr_unmix{ii} = corr_unmix;
            list_corr_unmix_active{ii} = corr_unmix_active;
            list_corr_unmix_inactive{ii} = corr_unmix_inactive;

            parfor kk=1:num_ratio
                thred_ratio=list_thred_ratio(kk);
    %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
    %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                    = GetPerformance_SpikeDetection_simulation_trace_split(...
                    output,traces_unmixed_filt,thred_ratio,sigma,mu);
%                     spikes_frames,lag,traces_unmixed_filt,thred_ratio,sigma,mu,cons,fs,decay);
                list_recall(ii,kk)=recall; 
                list_precision(ii,kk)=precision;
                list_F1(ii,kk)=F1;
            end
        %         fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('\b\b\b\b\b\b\b\b\b');

        %%
        save(sprintf('NAOMi\\scores_split_%s_%s_%sVideo_%s_compSigma%s.mat',method,simu_opt,video,sigma_from,baseline_std),...
            'list_recall','list_precision','list_F1','list_thred_ratio','list_corr_unmix',...
            'list_corr_unmix_active','list_corr_unmix_inactive');
        mean_F1 = squeeze(mean(list_F1,1));
        [max_F1, ind_max] = max(mean_F1(:));
        L1 = length(mean_F1);
        mean_corr = mean(cell2mat(list_corr_unmix));
        disp([list_thred_ratio(ind_max),max_F1,mean_corr])
        fprintf('\b');
        if ind_max == 1
            disp('Decrease thred_ratio');
        elseif ind_max == L1
            disp('Increase thred_ratio');
        end
        end
    end
%     fprintf('\b\b\b\b\b\b\b\b\b\b\b');
end
end
end
end
