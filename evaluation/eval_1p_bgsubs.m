clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));

%% choose video file
% dir_video='E:\OnePhoton videos\cropped videos\';
% dir_label = [dir_video,'split\'];
dir_video = '..\data\1p';
dir_traces='..\results\1p\unmixed traces\';
% dir_traces=dir_video;
dir_label = [dir_video,'\GT transients'];
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% list_Exp_ID = list_Exp_ID(1:5);
list_spike_type = {'1p'}; % {'only','include','exclude'};
% spike_type = 'exclude'; % {'include','exclude','only'};
sigma_from = 'Raw'; % {'Raw','Unmix'}; 
% video='Raw'; % {'Raw','SNR'}
list_video= {'Raw','SNR'};
list_baseline_std = {'_ksd-psd'}; % '', 
addon = ''; % '_eps=0.1'; % 
% dFF = h5read('E:\OnePhoton videos\1P_spike_tempolate.h5','/filter_tempolate')';
load('..\template\1P_spike_tempolate.mat','filter_tempolate');
dFF = squeeze(filter_tempolate)';
dFF = dFF(dFF>exp(-1));
dFF = dFF'/sum(dFF);
method = 'ours';
num_Exp=length(list_Exp_ID);


for bsid = 1:length(list_baseline_std)
    baseline_std = list_baseline_std{bsid};
% baseline_std = ''; % '_psd'; % '_ksd-psd'; % 
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
    for vid = 1:length(list_video)
        video = list_video{vid};
        if contains(baseline_std, 'psd')
            if contains(video,'SNR')
%                 list_thred_ratio=10:5:60; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                list_thred_ratio=100:10:200;% jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
            else
%                 list_thred_ratio=0:10; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                list_thred_ratio=100:10:200; % jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
            end
        else
            list_thred_ratio=4:12; % 6:0.5:9; % 8:2:20; % 
        end
        num_ratio=length(list_thred_ratio);
        folder = sprintf('traces_%s_%s%s',method,video,addon);
        dir_FISSA = fullfile(dir_traces,folder);
        useTF = strcmp(video, 'Raw');
        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));

        if useTF
            kernel=fliplr(dFF);
        else
            kernel = 1;
        end

        %%
        fprintf('Neuron Toolbox');
        for ii = 1:num_Exp
            Exp_ID = list_Exp_ID{ii};
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b%12s: ',Exp_ID);
            load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
            if strcmp(spike_type,'exclude')
                for oo = 1:length(output)
                    if ~isempty(output{oo}) && all(output{oo}(:,3))
                        output{oo}=[];
                    end
                end
            elseif strcmp(spike_type,'only')
                for oo = 1:length(output)
                    if ~isempty(output{oo}) && ~all(output{oo}(:,3))
                        output{oo}=[];
                    end
                end
            end

            load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
%             bgtraces = 0*bgtraces;
            if useTF
                traces_raw_filt=conv2(traces'-bgtraces',kernel,'valid');
            else
                traces_raw_filt=traces'-bgtraces';
            end
            [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

%             fprintf('Neuro Toolbox');
            traces_unmixed_filt = traces_raw_filt;
            [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);
            if strcmp(sigma_from,'Raw')
                sigma = sigma_raw;
            else
                sigma = sigma_unmixed;
            end

            parfor kk=1:num_ratio
                thred_ratio=list_thred_ratio(kk);
                [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                    = GetPerformance_SpikeDetection_split(output,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed);
                list_recall(ii,kk)=recall; 
                list_precision(ii,kk)=precision;
                list_F1(ii,kk)=F1;
            end
%             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
        %%
        if ~exist(spike_type)
            mkdir(spike_type);
        end
        save(sprintf('%s\\scores_split_bgsubs_%sVideo%s.mat',spike_type,video,baseline_std),...
            'list_recall','list_precision','list_F1','list_thred_ratio');
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
    end
end
end

%% merge Table_time for two or three trials
% dir_path = 'D:\ABO\20 percent 200\traces_ours_SNR_diag1_v2\';
% part1=load([dir_path,'Table_time (0.03-0.05).mat']);
% part2=load([dir_path,'Table_time (0.1-3).mat']);
% list_alpha = [double(part1.list_alpha),double(part2.list_alpha)];
% Table_time = [part1.Table_time(:,1:end-1), part2.Table_time(:,1:end)];
% save([dir_path,'Table_time.mat'],'list_alpha','Table_time');
