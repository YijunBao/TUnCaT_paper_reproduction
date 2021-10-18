clear;
addpath(genpath('..\evaluation'));

%% set parameters
% list_method = {'FISSA','ours'};
% list_method = {'diag_v1'};
% list_method = {'diag_v1','diag_v2'};
list_video = {'Raw'}; % 'SNR','Raw'
addon = '_fixed_alpha';
method = 'ours';
% list_alpha = [0.03, 0.1, 0.3, 1, 3];
% list_ind_alpha = 1:2:9;
% num_alpha = length(list_alpha);
% num_alpha = 13;
num_video = length(list_video);

spike_type = 'ABO'; % {'include','exclude','only'};
% dir_video='D:\ABO\20 percent 200';
% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master\BGsubs\split';
dir_video='..\data\ABO\';
dir_traces='..\results\ABO\unmixed traces\';
dir_label = [dir_video,'GT transients\'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};

baseline_std = '_ksd-psd';  % '' % 
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

% dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
load('..\template\GCaMP6f_spike_tempolate_mean.mat','filter_tempolate');
dFF = squeeze(filter_tempolate)';
dFF = dFF(dFF>exp(-1));
dFF = dFF'/sum(dFF);
[val,loc]=max(dFF);
lag = loc-1;

%% choose video file and and load manual spikes
ii = 3; % 2; % 6 for old
Exp_ID = list_Exp_ID{ii};
load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
if contains(spike_type,'exclude')
    for oo = 1:length(output)
        if ~isempty(output{oo}) && all(output{oo}(:,3))
            output{oo}=[];
        end
    end
elseif contains(spike_type,'only')
    for oo = 1:length(output)
        if ~isempty(output{oo}) && ~all(output{oo}(:,3))
            output{oo}=[];
        end
    end
end

%%
for vid = 1:length(list_video)
    video = list_video{vid};
    useTF = strcmp(video, 'Raw');
    if useTF
        kernel=fliplr(dFF);
    else
        kernel = 1;
    end

%     if strcmp(method,'FISSA')
%         sigma_from = 'Unmix';
%     else
        sigma_from = 'Raw';
%     end

    %% find optimal alpha and thred_ratio from saved results
    if strcmp(method,'FISSA')
        load(sprintf('%s\\scores_split_%s_%sVideo_%sSigma.mat',spike_type,method,video,sigma_from),...
            'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha'); %  (tol=1e-4, max_iter=%d)
        folder = sprintf('traces_%s_%s',method,video); %  (tol=1e-4, max_iter=%d)
    else
        load([spike_type,'\scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat'],...
            'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
        folder = ['traces_ours_',video,addon];
    end
    dir_FISSA = fullfile(dir_traces,folder);
    array_F1 = squeeze(list_F1(ii,:,:));
    
    num_alpha = length(list_alpha);
    if vid==1
        [recall,precision,F1] = deal(zeros(num_alpha,num_video));
        [individual_recall,individual_precision,spikes_GT_array,spikes_eval_array] = deal(cell(num_alpha,num_video));
        [list_traces_raw,list_traces_raw_filt]=deal(cell(1,num_video));
        [list_traces_unmixed,list_traces_unmixed_filt]=deal(cell(num_alpha,num_video));
    end
    
    %% Load the raw traces
    if strcmp(method,'FISSA')
        load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'raw_traces');
    else
        load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
        raw_traces=traces'-bgtraces';
    end
    if useTF
        traces_raw_filt=conv2(raw_traces,kernel,'valid');
    else
        traces_raw_filt=raw_traces;
    end
    list_traces_raw{vid} = raw_traces;
    list_traces_raw_filt{vid} = traces_raw_filt;
    [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

    %% Load the unmixed traces    
    for aid = 1:length(list_alpha)
        [F1_max,ind_thred_ratio] = max(array_F1(aid,:));
        alpha = list_alpha(aid);
        thred_ratio = list_thred_ratio(ind_thred_ratio);
    %         thred_ratio = min(8,list_thred_ratio(ind_thred_ratio));

        if strcmp(method,'FISSA')
            load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'unmixed_traces');
        else
            load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'traces_nmfdemix');
            unmixed_traces = traces_nmfdemix';
        end
        if useTF
            traces_unmixed_filt=conv2(unmixed_traces,kernel,'valid');
        else
            traces_unmixed_filt=unmixed_traces;
        end
        list_traces_unmixed{aid,vid} = unmixed_traces;
        list_traces_unmixed_filt{aid,vid} = traces_unmixed_filt;
        [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

        if strcmp(sigma_from,'Raw')
            sigma = sigma_raw;
        else
            sigma = sigma_unmixed;
        end

        [recall(aid,vid), precision(aid,vid), F1(aid,vid),individual_recall{aid,vid},...
            individual_precision{aid,vid},spikes_GT_array{aid,vid},spikes_eval_array{aid,vid}]...
            = GetPerformance_SpikeDetection_split(output,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed);
    end
end
%%
individual_recall_array = cell2mat(individual_recall(:)');
individual_precision_array = cell2mat(individual_precision(:)');
individual_F1 = cellfun(@(x,y) 2./(1./x+1./y),individual_recall,individual_precision, 'UniformOutput',false);
individual_F1_array = 2./(1./individual_recall_array+1./individual_precision_array);
figure; imagesc(individual_F1_array); colorbar;
