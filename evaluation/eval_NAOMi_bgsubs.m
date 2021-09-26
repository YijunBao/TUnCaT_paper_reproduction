clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
list_prot = {'GCaMP6f'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
% list_prot = {'GCaMP6s','jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
for pid = 1:length(list_prot) % [2,4,8,16,32,64] % 
    prot = list_prot{pid};
for fs = 30 % [3, 10, 100, 300] % 
% for T = [30, 50, 320, 1020] % 1100 % 
% for N = [50, 100, 300, 400] % 
% for power = [10, 20, 30, 50, 70, 150] % [1, 3, 100] % 
% for noise = [3, 10, 30, 50, 100] % 
if contains(prot,'6')
%     fs = 30; % [90,300] % 3,10,
    simu_opt = sprintf('120s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_%s',fs,prot); % 
%     simu_opt = sprintf('%ds_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_%s',T,prot); % 
%     simu_opt = sprintf('120s_30Hz_N=%d_100mW_noise10+23_NA0.8,0.6_%s',N,prot); % 
%     simu_opt = sprintf('120s_30Hz_N=200_%dmW_noise10+23_NA0.8,0.6_%s',power,prot); % 
%     simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23x%s_NA0.8,0.6_%s',num2str(noise),prot); % 
else
    fs = 3;
    simu_opt = sprintf('1100s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_%s',fs,prot); %
end
% simu_opt = sprintf('1100s_%dHz_N=200_40mW_noise10+23_NA0.8,0.6_%s',fs,prot); % 
% simu_opt = sprintf('110s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_%s',fs,prot); % 
% simu_opt_split = split(simu_opt,'_');

% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
dir_video = '..\data\NAOMi';
dir_traces='..\results\NAOMi\unmixed traces\';
% dir_traces=dir_video;
% list_alpha = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]; %
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
% list_thred_ratio=1:0.5:6; % 6:16; % 1:6; % 0.3:0.3:3; % 
max_alpha = inf;
addon = ''; % _bgsubs
% std_method = 'quantile-based std comp';  % comp
% std_method = 'psd';  % comp
% baseline_method = 'median'; % 'ksd';  % 
baseline_std = '_ksd-psd'; % ''; % '_psd'; % 
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

list_spike_type = {'NAOMi'}; % {'exclude','only','include'};
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'Raw'}; % {'Unmix','Raw'}; % 
% video='Raw'; % {'Raw','SNR'}
list_video= {'Raw','SNR'}; % 'Raw','SNR'

method = 'ours'; % {'FISSA','ours'}
bin_option = ''; % 'Mean';
% addon = '_DivideSigma';
% list_ndiag={'diag1', 'diag11', 'diag', 'diag02', 'diag22'}; % 
list_ndiag = {''};
% list_ndiag = {'_l1=0.0','_l1=0.2','_l1=0.8','_l1=1.0'}; 
% list_ndiag = {'_l1=1.0','_l1=0.0', '_l1=0.2','_l1=0.8'}; %,'_l1=0.8'
% list_vsub={''}; % ,'v2'
vsub=''; % ,'v2'
% vsub='_diag11_v1'; % ,'v2'


load(['..\template\filter_template 100Hz ',prot,'_ind_con=10.mat'],'template'); % _ind_con=10
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
% lag = ceil(fs*rise) + ceil(fs*rise*3); % 3;
cons = ceil(fs*decay*0.1);

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

load([dir_video,'\GT Masks\Traces_etc_',list_Exp_ID{1},'.mat'],'clean_traces')
folder = sprintf('traces_%s_%s%s%s%s',method,'SNR',addon,list_ndiag{1},vsub);
dir_FISSA = fullfile(dir_traces,folder);
load(fullfile(dir_FISSA,'raw',[list_Exp_ID{1},'.mat']),'traces');
length_kernel_py = size(clean_traces,2) - size(traces,1)+1;
length_diff = length_kernel_py - length(kernel);
if length_diff > 0
    kernel = padarray(kernel,[0,length_diff],'replicate','pre');
elseif length_diff < 0
    kernel = kernel(1-length_diff:end);
end
            
%%
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
%     for ind_ndiag = 1:length(list_ndiag)
    ndiag = list_ndiag{1};
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    if contains(prot,'6')
                        list_thred_ratio=10:5:60; % 2:12; % 20:10:120;% 0.3:0.3:3; % 1:0.5:6; % 
                    else
                        list_thred_ratio=50:10:150;% 10:2:30;  % 6:12; % 0.3:0.3:3; % 1:0.5:6; %
                    end
                else
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110;% 0:10; % 10:2:30; % 20:10:120; % 0.3:0.3:3; % 1:0.5:6; % 
                    else
                        list_thred_ratio=50:10:150;% 10:2:30; % 20:10:120; % 6:12; % 0.3:0.3:3; % 1:0.5:6; %
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
    %         folder = sprintf('traces_ours_%s (tol=1e-4, max_iter=%d)',lower(video),max_iter);
%             folder = sprintf('traces_%s_%s%s%s%s\\BinUnmix%s_%d',method,video,addon,ndiag,vsub,bin_option,nbin);
            folder = sprintf('traces_%s_%s%s%s%s',method,video,addon,ndiag,vsub);
    %         folder = sprintf('traces_ours');
            dir_FISSA = fullfile(dir_traces,folder);
            useTF = contains(video, 'Raw');

            num_Exp=length(list_Exp_ID);
            num_ratio=length(list_thred_ratio);
            [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));
            [list_corr_unmix,list_MSE_all,list_MSE_rmmean,list_MSE_rmmedian, ...
                list_pct_min,list_corr_unmix_active,list_corr_unmix_inactive]...
                =deal(cell(num_Exp, 1));
            list_corr_raw = cell(num_Exp,1);

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

                load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
                traces_raw=traces'-bgtraces';
                if useTF
                    traces_raw_filt=conv2(traces_raw,kernel,'valid');
                else
                    traces_raw_filt=traces_raw;
                end
%                 traces_raw_filt=traces_raw;
                traces_unmixed_filt = traces_raw_filt;
                [mu, sigma] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
                [corr_raw,corr_unmix_active,corr_unmix_inactive] = deal(zeros(ncells,1));
%                 clip_pct = prctile(traces_raw,84,2);
%                 clip_pct = median(traces_raw,2);
                for n = 1:ncells
%                     corr_raw(n) = corr(max(clip_pct(n),traces_raw(n,:)'),calcium(n,:)');
%                     corr_raw(n) = corr(max(mu(n),traces_raw(n,:)'),calcium(n,:)');
                    corr_raw(n) = corr(traces_raw(n,:)',calcium(n,:)');
                    corr_unmix_active(n) = corr(traces_raw(n,spikes_GT_line(n,:))',calcium(n,spikes_GT_line(n,:))');
                    corr_unmix_inactive(n) = corr(traces_raw(n,~spikes_GT_line(n,:))',calcium(n,~spikes_GT_line(n,:))');
                end
                list_corr_unmix{ii} = corr_raw;
                list_corr_unmix_active{ii} = corr_unmix_active;
                list_corr_unmix_inactive{ii} = corr_unmix_inactive;

                parfor kk=1:num_ratio
                    thred_ratio=list_thred_ratio(kk);
        %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
        %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                    [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                        = GetPerformance_SpikeDetection_simulation_trace_split(...
                        output,traces_unmixed_filt,thred_ratio,sigma,mu);
%                         spikes_frames,lag,traces_unmixed_filt,thred_ratio,sigma,mu,cons,fs,decay);
                    list_recall(ii,kk)=recall; 
                    list_precision(ii,kk)=precision;
                    list_F1(ii,kk)=F1;
                end
            %         fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
            end
            fprintf('\b\b\b\b\b\b\b\b\b');

            %%
            mean_F1 = squeeze(mean(list_F1,1));
            [max_F1, ind_max] = max(mean_F1(:));
            L1 = length(mean_F1);
            disp([list_thred_ratio(ind_max),max_F1])
            fprintf('\b');
            if ind_max == 1
                disp('Decrease thred_ratio');
            elseif ind_max == L1
                disp('Increase thred_ratio');
            end
            
            if ~exist(spike_type)
                mkdir(spike_type);
            end
%             list_corr_unmix_mean = [cellfun(@mean, list_corr_raw),cellfun(@mean, list_corr_unmix)];
            save(sprintf('NAOMi\\scores_split_%s_%s_%sVideo_%s_Sigma%s.mat','bgsubs',simu_opt,video,sigma_from,baseline_std),...
                'list_recall','list_precision','list_F1','list_thred_ratio','list_corr_unmix');
        end
    end
end
end
end