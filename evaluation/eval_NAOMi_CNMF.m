clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
list_prot = {'GCaMP6f'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
% list_prot = {'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
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
% simu_opt = '900s_30Hz_100+10\'; %_100+100
% dir_video=['F:\NAOMi\',simu_opt]; % _hasStart
dir_video = '..\data\NAOMi';
dir_traces='..\results\NAOMi\unmixed traces\';
% dir_traces=dir_video;
% list_alpha = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]; %
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
% list_thred_ratio=1:0.5:6; % 6:16; % 1:6; % 0.3:0.3:3; % 
max_alpha = inf;
% std_method = 'quantile-based std comp';  % comp
% std_method = 'psd';  % comp
% baseline_method = 'median'; % 'ksd';  % 
addon = ''; 
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
list_sigma_from = {'sum'}; % {'pure'}; % 
% video='Raw'; % {'Raw','SNR'}
list_video= {'Raw','SNR'}; % 'Raw','SNR'

method = 'CNMF'; % {'FISSA','ours','CNMF'}
% addon = '_DivideSigma';
% list_ndiag={'diag1', 'diag11', 'diag', 'diag02', 'diag22'}; % 
list_ndiag = {'_p1'}; % 
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
load(fullfile(dir_FISSA,[list_Exp_ID{1},'.mat']),'C_gt','YrA_gt');
length_kernel_py = size(clean_traces,2) - size(C_gt,2)+1;
length_diff = length_kernel_py - length(kernel);
if length_diff > 0
    kernel = padarray(kernel,[0,length_diff],'replicate','pre');
elseif length_diff < 0
    kernel = kernel(1-length_diff:end);
end
            
%%
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    if ~exist(spike_type)
        mkdir(spike_type);
    end
    for ind_ndiag = 1:length(list_ndiag)
        ndiag = list_ndiag{ind_ndiag};
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
                        list_thred_ratio=50:10:150;% jGCaMP7
                    end
                else
                    if contains(prot,'6')
                        list_thred_ratio=10:10:110; % GCaMP6f
%                         list_thred_ratio=80:10:180; % GCaMP6s
                    else
                        list_thred_ratio=50:10:150;% jGCaMP7
                    end
                end
            else
                if contains(prot,'6')
                    list_thred_ratio=0:0.5:5; % 6:16; % 
                else
                    list_thred_ratio=4:14; % 6:16; %
                end
            end
            if contains(prot,'6') % && contains(video,'SNR')
                list_thred_ratio = list_thred_ratio*sqrt(fs/30);
            end
    %         folder = sprintf('traces_ours_%s (tol=1e-4, max_iter=%d)',lower(video),max_iter);
            folder = sprintf('traces_%s_%s%s%s%s',method,video,addon,ndiag,vsub);
    %         folder = sprintf('traces_ours');
            dir_FISSA = fullfile(dir_traces,folder);
            useTF = contains(video, 'Raw');

            dir_sub = dir(dir_FISSA);

            num_Exp=length(list_Exp_ID);
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

                load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'C_gt','YrA_gt');
                traces_pure = C_gt;
                noise_pure = YrA_gt;
                ncells_remain = size(traces_pure,1);
                if ncells_remain < ncells
                    num_miss = ncells - size(traces_pure,1);
                    GTMasks_2_permute = reshape(permute(FinalMasks,[2,1,3]),[],ncells);
                    load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'A_thr');
                    if size(A_thr,2) ~= size(C_gt,1)
                        load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'A_gt');
                        A_thr = A_gt > max(A_gt,[],2)*0.2;
                    end
                    Dmat = JaccardDist_2(GTMasks_2_permute,A_thr);
                    ind_remove = zeros(1,num_miss);
%                     for ind = 1:num_miss
%                         diag_Dmat = diag(Dmat,-ind);
%                         diag_Dmat_smooth = movmean(diag_Dmat,3);
% %                         diff2 = diff(diag_Dmat_smooth,2);
%                         ind_remove(ind) = find(diag_Dmat_smooth<0.85,1)+1;
%                     end           
                    di = 0;
                    for ix = 1:ncells_remain
                        iy = ix + di;
                        while (iy < ncells) && (Dmat(iy,ix) >= Dmat(iy+1,ix))
                            di = di + 1;
                            ind_remove(di) = ix;
                            iy = iy + 1;
                        end
                    end
                    while iy < ncells
                        di = di + 1;
                        ind_remove(di) = ix;
                        iy = iy + 1;
                    end
%                     if di ~= num_miss
%                         error('Need manual selection');
%                     end
                    fp = fopen(sprintf('NAOMi\\removed neurons in %s Exp %s.txt',simu_opt,Exp_ID),'w');
                    fprintf(fp, 'Neuron %d is missing\n',ind_remove); 
                    fclose(fp);
                    T = size(C_gt,2);
                    empty_trace = zeros(1,T);
                    empty_noise = median(YrA_gt,1);
                    for ind = fliplr(ind_remove)
                        traces_pure = [traces_pure(1:ind-1,:);empty_trace;traces_pure(ind:end,:)];
                        noise_pure = [noise_pure(1:ind-1,:);empty_trace;noise_pure(ind:end,:)];
                    end
                end
                traces_sum = traces_pure+noise_pure;
%                 if ~useTF
%                     traces_pure=conv2(traces_pure,kernel,'valid');
%                     noise_pure=conv2(noise_pure,kernel,'valid');
%                     traces_sum=conv2(traces_sum,kernel,'valid');
%                 end
                [~, sigma_pure] = SNR_normalization(noise_pure,std_method,baseline_method);
                [mu_pure, ~] = SNR_normalization(traces_pure,std_method,baseline_method);
                [mu_sum, sigma_sum] = SNR_normalization(traces_sum,std_method,baseline_method);
                if useTF
                    noise_pure_filt=conv2(noise_pure,kernel,'valid');
                    traces_pure_filt=conv2(traces_pure,kernel,'valid');
                    traces_unmixed_filt=conv2(traces_sum,kernel,'valid');
                else
                    noise_pure_filt=noise_pure;
                    traces_pure_filt=traces_pure;
                    traces_unmixed_filt=traces_sum;
                end
                [~, sigma_pure_filt] = SNR_normalization(noise_pure_filt,std_method,baseline_method);
                [mu_pure_filt, ~] = SNR_normalization(traces_pure_filt,std_method,baseline_method);
                [mu_sum_filt, sigma_sum_filt] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

                [corr_unmix,corr_unmix_active,corr_unmix_inactive] = deal(zeros(ncells,1));
%                 clip_pct = prctile(traces_sum,84,2);
%                 clip_pct = median(traces_sum,2);
                for n = 1:ncells
%                     corr_unmix(n) = corr(max(clip_pct(n),traces_sum(n,:)'),calcium(n,:)');
%                     corr_unmix(n) = corr(max(mu_sum(n),traces_sum(n,:)'),calcium(n,:)');
                    corr_unmix(n) = corr(traces_sum(n,:)',calcium(n,:)');
                    corr_unmix_active(n) = corr(traces_sum(n,spikes_GT_line(n,:))',calcium(n,spikes_GT_line(n,:))');
                    corr_unmix_inactive(n) = corr(traces_sum(n,~spikes_GT_line(n,:))',calcium(n,~spikes_GT_line(n,:))');
                end
                corr_unmix(isnan(corr_unmix)) = 0;
                corr_unmix(isnan(corr_unmix))=0;
                corr_unmix_active(isnan(corr_unmix_active))=0;
                corr_unmix_inactive(isnan(corr_unmix_inactive))=0;
                list_corr_unmix{ii} = corr_unmix;
                list_corr_unmix_active{ii} = corr_unmix_active;
                list_corr_unmix_inactive{ii} = corr_unmix_inactive;

                if strcmp(sigma_from,'pure')
                    sigma = sigma_pure_filt;
                    mu = mu_pure_filt;
                else
                    sigma = sigma_sum_filt;
                    mu = mu_sum_filt;
                end

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
%                 fprintf('\b\b\b\b\b\b\b');
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
            
%             list_corr_unmix_mean = [cellfun(@mean, list_corr_raw),cellfun(@mean, list_corr_unmix)];
            save(sprintf('NAOMi\\scores_split_%s_%s_%sVideo%s_%sSigma%s.mat',method,simu_opt,video,ndiag,sigma_from,baseline_std),...
                'list_recall','list_precision','list_F1','list_thred_ratio','list_corr_unmix');
        end
    end
    end
end
end
end