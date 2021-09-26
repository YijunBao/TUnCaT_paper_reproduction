clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
% dir_video='D:\ABO\20 percent 200';
% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master'; % 
dir_video = '..\data\ABO';
dir_traces='..\results\ABO\unmixed traces\';
% dir_traces=dir_video;
dir_label = [dir_video,'\GT transients'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
num_Exp=length(list_Exp_ID);

list_spike_type = {'ABO'}; % 
% list_spike_type = {'include','only','exclude'}; % 
% list_spike_type = cellfun(@(x) [x,'_noBGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'}; % 
list_baseline_std = {'_ksd-psd'}; % '', 
% std_method = 'quantile-based std comp';  % comp
MovMedianSubs = contains(list_spike_type{1},'MovMedian'); % false;

method = 'AllenSDK'; % {'FISSA','ours','CNMF','AllenSDK'}
list_video={'Raw','SNR'}; % 
addon = '';

% list_thred_ratio=6:0.5:9; % 6:12; % 9:16; % 
% num_ratio=length(list_thred_ratio);

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
                    list_thred_ratio=30:10:130; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                else
                    list_thred_ratio=30:10:130; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                end
            else
                list_thred_ratio=5:0.5:9; % 6:12; % 9:16; % 
            end
            folder = sprintf('traces_%s_%s%s%s%s%s',method,video,addon);
            dir_FISSA = fullfile(dir_traces,folder);
            useTF = strcmp(video, 'Raw');
            num_ratio=length(list_thred_ratio);

        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));

        if useTF
%             dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
            load('..\template\GCaMP6f_spike_tempolate_mean.mat','filter_tempolate');
            dFF = squeeze(filter_tempolate)';
            dFF = dFF(dFF>exp(-1));
            dFF = dFF'/sum(dFF);
            kernel=fliplr(dFF);
        else
            kernel = 1;
        end

        %%
        fprintf('Neuro Tools');
        for ii = 1:num_Exp
            Exp_ID = list_Exp_ID{ii};
            fprintf('\b\b\b\b\b\b\b\b\b\b\b%s: ',Exp_ID);
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

            load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'roi_traces','compensated_traces'); % ,'unmixed_traces'
            raw_traces = roi_traces;
            unmixed_traces = compensated_traces;
            if MovMedianSubs
                raw_traces = raw_traces - movmedian(raw_traces,900,2);
                unmixed_traces = unmixed_traces - movmedian(unmixed_traces,900,2);
            end
            if useTF
                traces_raw_filt=conv2(raw_traces,kernel,'valid');
                traces_unmixed_filt=conv2(unmixed_traces,kernel,'valid');
            else
                traces_raw_filt=raw_traces;
                traces_unmixed_filt=unmixed_traces;
            end
            [~, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
            [mu, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);
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
                    = GetPerformance_SpikeDetection_split(output,traces_unmixed_filt,thred_ratio,sigma,mu);
                list_recall(ii,kk)=recall;  % 
                list_precision(ii,kk)=precision;
                list_F1(ii,kk)=F1;
            end
%             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('\b\b\b\b\b\b\b\b\b\b\b');

        %         if num_ratio>1
        %             figure; plot(list_thred_ratio,[list_recall,list_precision,list_F1],'LineWidth',2);
        %             legend('Recall','Precision','F1');
        %             xlabel('thred_ratio','Interpreter','none');
        %             [~,ind]=max(list_F1);
        %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',list_recall(ind),list_precision(ind),list_F1(ind),list_thred_ratio(ind));
        %         else
        %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',recall, precision, F1,thred_ratio);
        %         end
        %%
        if ~exist(spike_type)
            mkdir(spike_type);
        end
        save(sprintf('%s\\scores_split_%s_%sVideo%s_%s%s.mat',spike_type,method,video,addon,sigma_from,baseline_std),...
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
%     fprintf('\b\b\b\b\b\b\b\b\b\b\b');
end
end