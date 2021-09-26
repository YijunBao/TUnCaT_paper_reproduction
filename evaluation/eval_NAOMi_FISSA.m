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

% simu_opt = sprintf('100s_%dHz_100+10_old',fs); %_100+100
% simu_opt = '900s_30Hz_100+10\'; %_100+100
% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
dir_video = '..\data\NAOMi';
dir_traces='..\results\NAOMi\unmixed traces\';
% dir_traces=dir_video;
% list_alpha = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]; %
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
% list_thred_ratio=1:0.5:6; % 6:16; % 1:6; % 0.3:0.3:3; % 
max_alpha = inf;

addon = ''; % _n_iter
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
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'}; % 
% video='Raw'; % {'Raw','SNR'}
list_video= {'Raw','SNR'}; % 'Raw','SNR'

method = 'FISSA'; % {'FISSA','ours'}
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
folder = sprintf('traces_%s_%s%s%s%s',method,'SNR',addon,list_ndiag{1},vsub);
dir_FISSA = fullfile(dir_traces,folder);
load(fullfile(dir_FISSA,'raw',[list_Exp_ID{1},'.mat']),'raw_traces');
length_kernel_py = size(clean_traces,2) - size(raw_traces,2)+1;
length_diff = length_kernel_py - length(kernel);
if length_diff > 0
    kernel = padarray(kernel,[0,length_diff],'replicate','pre');
elseif length_diff < 0
    kernel = kernel(1-length_diff:end);
end
            
%%
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
%     ind_ndiag = 1:length(list_ndiag);
    ndiag = list_ndiag{1};
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110; % GCaMP6f
%                         list_thred_ratio=80:10:180; % GCaMP6s
                    else
                        list_thred_ratio=50:10:150;% 10:2:30;  % 6:12; % 0.3:0.3:3; % 1:0.5:6; % 
                    end
                else
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110;% 0:10; % GCaMP6f
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
            folder = sprintf('traces_%s_%s%s%s%s',method,video,addon,ndiag,vsub);
    %         folder = sprintf('traces_ours');
            dir_FISSA = fullfile(dir_traces,folder);
            useTF = contains(video, 'Raw');

            dir_sub = dir(dir_FISSA);
            num_alpha = length(dir_sub);
            list_alpha = zeros(num_alpha,1);
            for aid = 1:num_alpha
                alpha_folder = dir_sub(aid);
                alpha_name = alpha_folder.name;
                if contains(alpha_name,'alpha')
                    alpha = split(alpha_name,'=');
                    alpha = str2double(alpha{2});
                    list_alpha(aid) = alpha;
                end
            end
            dir_sub(list_alpha==0)=[];
            list_alpha(list_alpha==0)=[];
            list_alpha = sort(list_alpha);
            list_alpha = list_alpha(list_alpha<=max_alpha);

            num_Exp=length(list_Exp_ID);
            num_alpha=length(list_alpha);
            num_ratio=length(list_thred_ratio);
            [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_alpha, num_ratio));
            [list_corr_unmix,list_MSE_all,list_MSE_rmmean,list_MSE_rmmedian, ...
                list_pct_min,list_corr_unmix_active,list_corr_unmix_inactive]...
                =deal(cell(num_Exp, num_alpha));
            list_corr_raw = cell(num_Exp,1);

            %%
            fprintf('Neuro Tools');
            for ii = 1:num_Exp
                Exp_ID = list_Exp_ID{ii};
                fprintf('\b\b\b\b\b\b\b\b\b\b\b%s: ',Exp_ID);
                load([dir_video,'\GT Masks\FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
                load([dir_video,'\GT Masks\Traces_etc_',Exp_ID,'.mat'],'spikes','clean_traces')
                ncells = size(FinalMasks,3);
%                 spikes = spikes(1:ncells)';
                calcium = clean_traces;
                spikes_cell = mat2cell(spikes,ones(ncells,1),size(spikes,2));
                spikes_frames = cellfun(@(x) find(x)*fs/100, spikes_cell, 'UniformOutput', false);
                if useTF
                    calcium_filt = conv2(calcium,kernel,'valid');
                else
                    calcium = conv2(calcium,kernel,'valid');
                    calcium_filt = calcium;
                end
%                 calcium = cell2mat(calcium);
                [output, spikes_GT_line] = GT_transient_NAOMi_split(calcium_filt,spikes_frames);

                load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'raw_traces');
                traces_raw=raw_traces;
                if useTF
                    traces_raw_filt=conv2(traces_raw,kernel,'valid');
                else
                    traces_raw_filt=traces_raw;
                end
                [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
                corr_raw = zeros(ncells,1);
%                 clip_pct = prctile(traces_raw,84,2);
%                 clip_pct = median(traces_raw,2);
                for n = 1:ncells
%                     corr_raw(n) = corr(max(clip_pct(n),traces_raw(n,:)'),calcium(n,:)');
%                     corr_raw(n) = corr(max(mu_raw(n),traces_raw(n,:)'),calcium(n,:)');
                    corr_raw(n) = corr(traces_raw(n,:)',calcium(n,:)');
                end
                list_corr_raw{ii} = corr_raw;

                fprintf('Neuron Toolbox');
                for jj = 1:num_alpha
                    alpha = list_alpha(jj);
                    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\balpha=%6.3f: ',alpha);
                    load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'unmixed_traces'); %
                    traces_nmfdemix = unmixed_traces;
%                     traces_unmixed_filt=traces_nmfdemix;
                    if useTF
                        traces_unmixed_filt=conv2(traces_nmfdemix,kernel,'valid');
                    else
                        traces_unmixed_filt=traces_nmfdemix;
                    end
                    [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

                    pct_min = mean(abs(traces_nmfdemix - min(traces_nmfdemix,[],2)) < eps('single'),2);
                    list_pct_min{ii,jj}=pct_min;
%                     list_MSE_all{ii,jj}=list_MSE;
%                     MSE_native = cellfun(@(x) x(1), list_MSE);
                    trace_diff = calcium-traces_nmfdemix;
                    trace_diff = trace_diff - mean(trace_diff,2);
                    MSE = mean((trace_diff).^2,2);
                    list_MSE_rmmean{ii,jj}=MSE;
                    trace_diff = trace_diff - median(trace_diff,2);
                    MSE = mean((trace_diff).^2,2);
                    list_MSE_rmmedian{ii,jj}=MSE;
                    
                    [corr_unmix,corr_unmix_active,corr_unmix_inactive] = deal(zeros(ncells,1));
%                     clip_pct = prctile(traces_nmfdemix,84,2);
%                     clip_pct = median(traces_nmfdemix,2);
                    for n = 1:ncells
%                         corr_unmix(n) = corr(max(clip_pct(n),traces_nmfdemix(n,:)'),calcium(n,:)');
%                         corr_unmix(n) = corr(max(mu_unmixed(n),traces_nmfdemix(n,:)'),calcium(n,:)');
                        corr_unmix(n) = corr(traces_nmfdemix(n,:)',calcium(n,:)');
                        corr_unmix_active(n) = corr(traces_nmfdemix(n,spikes_GT_line(n,:))',calcium(n,spikes_GT_line(n,:))');
                        corr_unmix_inactive(n) = corr(traces_nmfdemix(n,~spikes_GT_line(n,:))',calcium(n,~spikes_GT_line(n,:))');
                    end
                    corr_unmix(isnan(corr_unmix)) = 0;
                    list_corr_unmix{ii,jj} = corr_unmix;
                    list_corr_unmix_active{ii,jj} = corr_unmix_active;
                    list_corr_unmix_inactive{ii,jj} = corr_unmix_inactive;
                    
                    if strcmp(sigma_from,'Raw')
                        sigma = sigma_raw;
                    else
                        sigma = sigma_unmixed;
                    end

                    parfor kk=1:num_ratio
                        thred_ratio=list_thred_ratio(kk);
            %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
            %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                        [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                            = GetPerformance_SpikeDetection_simulation_trace_split(...
                            output,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed);
%                             spikes_frames,lag,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed,cons,fs,decay);
                        list_recall(ii,jj,kk)=recall; 
                        list_precision(ii,jj,kk)=precision;
                        list_F1(ii,jj,kk)=F1;
                    end
            %         fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
                end
                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b');
                if alpha >=100
                    fprintf('\b');
                end                    
            end
            fprintf('\b\b\b\b\b\b\b\b\b\b\b');

            %%
            mean_F1 = squeeze(mean(list_F1,1));
            [max_F1, ind_max] = max(mean_F1(:));
            [L1, L2] = size(mean_F1);
            [ind1, ind2] = ind2sub([L1, L2],ind_max);
            disp([list_alpha(ind1), list_thred_ratio(ind2),max_F1])
            fprintf('\b');
            if ind1 == 1
                disp('Decrease alpha');
            elseif ind1 == L1
                disp('Increase alpha');
            end
            if ind2 == 1
                disp('Decrease thred_ratio');
            elseif ind2 == L2
                disp('Increase thred_ratio');
            end
            
            if ~exist(spike_type)
                mkdir(spike_type);
            end
%             list_corr_unmix_mean = [cellfun(@mean, list_corr_raw),cellfun(@mean, list_corr_unmix)];
            save(sprintf('NAOMi\\scores_split_%s_%s_%sVideo_%s_Sigma%s%s.mat',method,simu_opt,video,sigma_from,addon,baseline_std),...
                'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha',...
                'list_corr_unmix','list_MSE_all','list_MSE_rmmean','list_MSE_rmmedian','list_pct_min');
        end
    end
end
end
end