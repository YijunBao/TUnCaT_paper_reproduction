clear;
addpath(genpath('..\evaluation'));

%% set parameters
% list_method = {'FISSA','ours'};
% list_method = {'diag_v1'};
% list_method = {'diag_v1','diag_v2'};
% list_video = {'SNR','Raw'}; % 
% alpha=10;
list_video = {'Raw'}; % {'SNR','Raw'}; 
num_video = length(list_video);
% list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'}; % ; 'Allen SDK'
list_method = {'BG subtraction'; 'FISSA'; 'CNMF'; 'AllenSDK'; 'TUnCaT'}; % ; 'Allen SDK'
% list_method = {'BG subtraction'; 'FISSA'; 'TUnCaT'; 'CNMF'}; % ; 'Allen SDK'
num_method = length(list_method);
% std_method = 'quantile-based std comp';  % comp
% std_method = 'psd';  % comp
addon = ''; 

spike_type = 'ABO'; % {'include','exclude','only'};
% max_iter = 20000;
% dir_video='D:\ABO\20 percent 200';
% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
dir_video='..\data\ABO\';
% dir_traces=dir_video;
% dir_scores='..\evaluation\ABO\';
dir_traces='..\results\ABO\unmixed traces\';
dir_scores='..\results\ABO\evaluation\';
dir_label = [dir_video,'GT transients\'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
MovMedianSubs = contains(spike_type,'MovMedian'); % false;

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

[recall,precision,F1] = deal(zeros(num_method,num_video));
[individual_recall,individual_precision,spikes_GT_array,spikes_eval_array] = deal(cell(num_method,num_video));
[list_traces_raw,list_traces_raw_filt]=deal(cell(1,num_video));
[list_traces_unmixed,list_traces_unmixed_filt]=deal(cell(num_method,num_video));

%% choose video file and and load manual spikes
ii = 4;
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
output_select = find(cellfun(@(x) ~isempty(x), output));

%%
for vid = 1:length(list_video)
    video = list_video{vid};
    useTF = strcmp(video, 'Raw');
    if useTF
        kernel=fliplr(dFF);
    else
        kernel = 1;
    end
%     sigma_from = 'Raw';
    list_scorefile = {['scores_split_bgsubs_',video,'Video',baseline_std,'.mat'],...
        ['scores_split_FISSA_',video,'Video_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_CNMF_',video,'Video_p1_sum',baseline_std,'.mat'],...
        ['scores_split_AllenSDK_',video,'Video_Unmix',baseline_std,'.mat'],...
        ['scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat']...
        }'; %  (tol=1e-4, max_iter=20000)
%         ['scores_',video,'Video_traces_ours_',video,'_sigma1_diag11_v1_RawSigma.mat'],...
%     list_scorefile{1}=list_scorefile{3};
%     list_tracefile = {['traces_ours_',video,'_sigma1_diag11_v1'],...
    list_tracefile = {['traces_ours_',video,'',addon],...
        ['traces_FISSA_',video,''],...
        ['traces_CNMF_',video,'_p1'],...
        ['traces_AllenSDK_',video],...
        ['traces_ours_',video,addon]...
        }'; %  (tol=1e-4, max_iter=20000)
%     list_tracefile{1}=list_tracefile{3};

    for mid = 1:length(list_method)
%         method = list_method(mid);
        %% find optimal alpha and thred_ratio from saved results
        load(fullfile(dir_scores,list_scorefile{mid}),'list_recall','list_precision','list_F1','list_alpha','list_thred_ratio');
        folder = list_tracefile{mid};
        dir_FISSA = fullfile(dir_traces,folder);
        array_F1 = squeeze(list_F1(ii,:,:));

        %% Load the raw traces
        if mid==1 % raw
            load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
            raw_traces=traces'-bgtraces';
            if MovMedianSubs
                raw_traces = raw_traces - movmedian(raw_traces,900,2);
            end
            if useTF
                traces_raw_filt=conv2(raw_traces,kernel,'valid');
            else
                traces_raw_filt=raw_traces;
            end
            list_traces_raw{vid} = raw_traces;
            list_traces_raw_filt{vid} = traces_raw_filt;
            [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
            list_traces_unmixed{mid,vid} = raw_traces;
            list_traces_unmixed_filt{mid,vid} = traces_raw_filt;

            if size(array_F1,2)~=1
                array_F1=array_F1';
            end
            [F1_max,ind] = max(array_F1(:));
            [ind_thred_ratio,aid] = ind2sub(size(array_F1),ind);
            thred_ratio = list_thred_ratio(ind_thred_ratio);

            [recall(mid,vid), precision(mid,vid), F1(mid,vid),individual_recall{mid,vid},...
                individual_precision{mid,vid},spikes_GT_array{mid,vid},spikes_eval_array{mid,vid}]...
                = GetPerformance_SpikeDetection_split(output,traces_raw_filt,thred_ratio,sigma_raw,mu_raw);
        
%         elseif mid==3 % percent pixels
%             list_traces_unmixed_filt{mid,vid} = traces_raw_filt;
%             saved_result = load(fullfile(spike_type,list_scorefile{mid}),'individual_recall','individual_precision',...
%                 'spikes_GT_array','spikes_eval_array');
%             array_F1 = squeeze(list_F1(ii,:,:));
%             if size(array_F1,2)~=1
%                 array_F1=array_F1';
%             end
%             [F1_max,ind] = max(array_F1(:));
%             [ind_thred_ratio,aid] = ind2sub(size(array_F1),ind);
%             recall(mid,vid)=list_recall(ind_thred_ratio,aid);
%             precision(mid,vid)=list_precision(ind_thred_ratio,aid);
%             F1(mid,vid)=list_F1(ind_thred_ratio,aid);
%             individual_recall{mid,vid}=saved_result.individual_recall{ii,ind_thred_ratio};
%             individual_precision{mid,vid}=saved_result.individual_precision{ii,ind_thred_ratio};
%             spikes_GT_array{mid,vid}=saved_result.spikes_GT_array{ii,ind_thred_ratio};
%             spikes_eval_array{mid,vid}=saved_result.spikes_eval_array{ii,ind_thred_ratio};
        
        elseif mid==3 % CNMF
            array_F1 = squeeze(list_F1(ii,:));
            if size(array_F1,2)~=1
                array_F1=array_F1';
            end
            [F1_max,ind] = max(array_F1(:));
            [ind_thred_ratio,aid] = ind2sub(size(array_F1),ind);
            thred_ratio = list_thred_ratio(ind_thred_ratio);

            load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'C_gt','YrA_gt');
            traces_pure = C_gt;
            noise_pure = YrA_gt;
            traces_sum = C_gt+YrA_gt;
            if MovMedianSubs
                traces_sum = traces_sum - movmedian(traces_sum,900,2);
            end
            if useTF
                traces_pure_filt=conv2(traces_pure,kernel,'valid');
                noise_pure_filt=conv2(noise_pure,kernel,'valid');
                traces_sum_filt=conv2(traces_sum,kernel,'valid');
            else
                traces_pure_filt=traces_pure;
                noise_pure_filt=noise_pure;
                traces_sum_filt=traces_sum;
            end
            [~, sigma_pure] = SNR_normalization(noise_pure_filt,std_method,baseline_method);
            [mu_pure, ~] = SNR_normalization(traces_pure_filt,std_method,baseline_method);
            [mu_sum, sigma_sum] = SNR_normalization(traces_sum_filt,std_method,baseline_method);
%             if strcmp(sigma_from,'pure')
%                 sigma = sigma_pure;
%                 mu = mu_pure;
%                 traces_filt = traces_pure;
%             else
                sigma = sigma_sum;
                mu = mu_sum;
                traces_filt = traces_sum_filt;
%             end
%             if useTF
%                 traces_filt=conv2(traces_filt,kernel,'valid');
%             else
%                 traces_filt=traces_filt;
%             end
            list_traces_unmixed{mid,vid} = traces_sum;
            list_traces_unmixed_filt{mid,vid} = traces_sum_filt;

%             recall(mid,vid)=list_recall(ind_thred_ratio,aid);
%             precision(mid,vid)=list_precision(ind_thred_ratio,aid);
%             F1(mid,vid)=list_F1(ind_thred_ratio,aid);
%             individual_recall{mid,vid}=saved_result.individual_recall{ii,ind_thred_ratio};
%             individual_precision{mid,vid}=saved_result.individual_precision{ii,ind_thred_ratio};
%             spikes_GT_array{mid,vid}=saved_result.spikes_GT_array{ii,ind_thred_ratio};
%             spikes_eval_array{mid,vid}=saved_result.spikes_eval_array{ii,ind_thred_ratio};

            [recall(mid,vid), precision(mid,vid), F1(mid,vid),individual_recall{mid,vid},individual_precision{mid,vid},spikes_GT_array{mid,vid},spikes_eval_array{mid,vid}]...
                = GetPerformance_SpikeDetection_split(output,traces_filt,thred_ratio,sigma,mu);
        
        else% if any(mid==[2,3,5])
            %% Load the unmixed traces
            array_F1 = squeeze(list_F1(ii,:,:));
            if size(array_F1,2)~=1
                array_F1=array_F1';
            end
            [F1_max,ind] = max(array_F1(:));
            [ind_thred_ratio,aid] = ind2sub(size(array_F1),ind);
            alpha = list_alpha(aid);
            thred_ratio = list_thred_ratio(ind_thred_ratio);

            if mid==2
                load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'unmixed_traces');
            elseif mid==5
                load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'traces_nmfdemix');
                unmixed_traces = traces_nmfdemix';
            elseif mid==4 % AllenSDK
                load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'compensated_traces');
                unmixed_traces = compensated_traces;
            end
            
            if MovMedianSubs
                unmixed_traces = unmixed_traces - movmedian(unmixed_traces,900,2);
            end
            if useTF
                traces_unmixed_filt=conv2(unmixed_traces,kernel,'valid');
            else
                traces_unmixed_filt=unmixed_traces;
            end
            list_traces_unmixed{mid,vid} = unmixed_traces;
            list_traces_unmixed_filt{mid,vid} = traces_unmixed_filt;
            [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

            if mid==1
                sigma = sigma_raw;
            else % mid==2,4
                sigma = sigma_unmixed;
            end

            [recall(mid,vid), precision(mid,vid), F1(mid,vid),individual_recall{mid,vid},...
                individual_precision{mid,vid},spikes_GT_array{mid,vid},spikes_eval_array{mid,vid}]...
                = GetPerformance_SpikeDetection_split(output,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed);
        end
        fprintf('%.4f\n',F1_max)
    end
end
%%
individual_recall_array = cell2mat(individual_recall(:)');
individual_precision_array = cell2mat(individual_precision(:)');
individual_F1 = cellfun(@(x,y) 2./(1./x+1./y),individual_recall,individual_precision, 'UniformOutput',false);
individual_F1_array = 2./(1./individual_recall_array+1./individual_precision_array);

