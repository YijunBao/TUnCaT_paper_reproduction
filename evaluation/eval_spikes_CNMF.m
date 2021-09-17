clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
list_baseline_std = {'_ksd-psd'}; % '', 
list_spike_type = {'only_BGSubs'}; % 
% list_spike_type = {'include','only','exclude'}; % 
% list_spike_type = cellfun(@(x) [x,'_noBGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'sum'}; % {'pure','sum'}; 

method = 'CNMF'; % {'FISSA','ours','CNMF'}
list_video={'Raw','SNR'};
% video='Raw'; % {'Raw','SNR'}
addon = '';
% list_ndiag={'diag1', 'diag11', 'diag', 'diag02', 'diag22'}; % 
list_ndiag = {'_p1','_p2'}; %, '_diag11'
% list_ndiag = {'+1-3', '+3-0', '+3-1'}; %'diag01', 
% list_ndiag = {'_l1=1.0','_l1=0.0', '_l1=0.2','_l1=0.8'}; %,'_l1=0.8'

dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
dir_video='D:\ABO\20 percent 200';
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
num_Exp=length(list_Exp_ID);
MovMedianSubs = contains(list_spike_type{1},'MovMedian'); % false;

% baseline_std = '_ksd-psd'; % ''; % '_psd'; % 
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

%%
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    for ind_ndiag = 1:length(list_ndiag)
        ndiag = list_ndiag{ind_ndiag};
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
        if contains(baseline_std, 'psd')
            if contains(video,'SNR')
                list_thred_ratio=50:5:100; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
            else
                list_thred_ratio=50:5:100; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
            end
        else
            list_thred_ratio=6:0.5:9; % 6:12; % 9:16; %
        end
        num_ratio=length(list_thred_ratio);
%         folder = sprintf('traces_ours_%s (tol=1e-4, max_iter=%d)',lower(video),max_iter);
        folder = sprintf('traces_%s_%s%s%s',method,video,addon,ndiag);
%         folder = sprintf('traces_ours');
        dir_FISSA = fullfile(dir_video,folder);
        useTF = strcmp(video, 'Raw');

        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));

        if useTF
            dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
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

            load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'C_gt','YrA_gt');
            traces_pure = C_gt;
            noise_pure = YrA_gt;
            traces_sum = C_gt+YrA_gt;
            if MovMedianSubs
                traces_sum = traces_sum - movmedian(traces_sum,900,2);
            end
            if useTF
                traces_pure=conv2(traces_pure,kernel,'valid');
                noise_pure=conv2(noise_pure,kernel,'valid');
                traces_sum=conv2(traces_sum,kernel,'valid');
            end
            [~, sigma_pure] = SNR_normalization(noise_pure,std_method,baseline_method);
            [mu_pure, ~] = SNR_normalization(traces_pure,std_method,baseline_method);
            [mu_sum, sigma_sum] = SNR_normalization(traces_sum,std_method,baseline_method);

%                 fprintf('Neuron Toolbox');
            if strcmp(sigma_from,'pure')
                sigma = sigma_pure;
                mu = mu_pure;
                traces_filt = traces_pure;
            else
                sigma = sigma_sum;
                mu = mu_sum;
                traces_filt = traces_sum;
            end

            parfor kk=1:num_ratio
                thred_ratio=list_thred_ratio(kk);
    %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
    %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                    = GetPerformance_SpikeDetection_split(output,traces_filt,thred_ratio,sigma,mu);
                list_recall(ii,kk)=recall; 
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
        save(sprintf('%s\\scores_split_%s_%sVideo%s%s_%s%s.mat',spike_type,method,video,addon,ndiag,sigma_from,baseline_std),...
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
end