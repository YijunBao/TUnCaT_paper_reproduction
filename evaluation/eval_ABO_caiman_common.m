clear;
% addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))
% addpath(genpath('C:\Other methods\caiman_data'))
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
list_baseline_std = {'_ksd-psd'}; % '', 
list_spike_type = {'ABO'}; % 
% list_spike_type = {'include','only','exclude'}; % 
% list_spike_type = cellfun(@(x) [x,'_noBGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'sum'}; % {'pure','sum'}; 
% onlyTP = true;

method = 'CaImAn'; % {'FISSA','ours','CNMF'}
list_video={'Raw','SNR'};
% video='Raw'; % {'Raw','SNR'}
addon = '';
% list_ndiag={'diag1', 'diag11', 'diag', 'diag02', 'diag22'}; % 
list_ndiag = {''}; %, '_diag11'
% list_ndiag = {'+1-3', '+3-0', '+3-1'}; %'diag01', 
% list_ndiag = {'_l1=1.0','_l1=0.0', '_l1=0.2','_l1=0.8'}; %,'_l1=0.8'
% list_IoU = 0.2:0.1:0.5; % 
ThJaccard = 0.5;

% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
% dir_video='D:\ABO\20 percent 200';
dir_video = '..\data\ABO';
dir_traces='..\results\ABO\unmixed traces\';
% dir_traces=dir_video;
dir_label = [dir_video,'\GT transients'];
dir_GTMasks = [dir_video,'\GT Masks'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
num_Exp=length(list_Exp_ID);
MovMedianSubs = contains(list_spike_type{1},'MovMedian'); % false;
% [SpatialRecall, SpatialPrecision, SpatialF1] = deal(zeros(1,10));
% m = cell(1,10);

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
    % for ind_IoU = 1:length(list_IoU)
    %     ThJaccard = list_IoU(ind_IoU);
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
%         folder = sprintf('traces_%s_%s%s%s',method,video,addon,ndiag);
        folder_SUNS = 'SUNS_complete Masks';
        dir_SUNS = fullfile(dir_traces,folder_SUNS);
        folder_caiman = sprintf('caiman-Batch_%s\\275',video);
        dir_caiman = fullfile(dir_traces,folder_caiman);
        useTF = strcmp(video, 'Raw');

        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));

        if useTF
            % dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
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
            load(fullfile(dir_SUNS,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
            Masks_SUNS = FinalMasks;
            load(fullfile(dir_caiman,[Exp_ID,'.mat']),'Ab','C','YrA'); % 
            L1=sqrt(size(Ab,1));
            Masks_caiman = ProcessOnACIDMasks(Ab,[L1,L1],0.2);  
            
            [~, ~, ~,m_GT_SUNS] = GetPerformance_Jaccard(...
                dir_GTMasks,Exp_ID,Masks_SUNS,ThJaccard);
            [~, ~, ~,m_GT_caiman] = GetPerformance_Jaccard(...
                dir_GTMasks,Exp_ID,Masks_caiman,ThJaccard);
            [~, ~, ~,m_SUNS_caiman] = GetPerformance_Jaccard(...
                dir_SUNS,Exp_ID,Masks_caiman,ThJaccard);
            [select_SUNS_GT, select_GT_SUNS] = find(m_GT_SUNS');
            [select_caiman_GT, select_GT_caiman] = find(m_GT_caiman');
            [select_caiman_SUNS, select_SUNS_caiman] = find(m_SUNS_caiman');
            select_GT = intersect(select_GT_SUNS, select_GT_caiman);
            num_common = length(select_GT);
            commons = zeros(num_common,3);
            commons(:,1) = select_GT;
            select_GT_use = logical(select_GT);
            for ind = 1:num_common
                ind_caiman=select_caiman_GT(find(select_GT_caiman==select_GT(ind)));
                commons(ind,3)=ind_caiman;
                ind_SUNS=select_SUNS_GT(find(select_GT_SUNS==select_GT(ind)));
                commons(ind,2)=ind_SUNS;
                select_GT_use(ind) = m_SUNS_caiman(ind_SUNS,ind_caiman);
            end
            commons = commons(select_GT_use,:);
            select_GT = commons(:,1);
            select_SUNS = commons(:,2);
            select_caiman = commons(:,3);
            
            load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
            output = output(select_GT);
            C_gt = C(select_caiman,:);
            YrA_gt = YrA(select_caiman,:);
            
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
                    = GetPerformance_SpikeDetection_FPneurons(output,traces_filt,thred_ratio,sigma,mu);
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
%         SpatialF1
        if ~exist(spike_type)
            mkdir(spike_type);
        end
        save(sprintf('%s\\scores_split_%s_%sVideo%s_%s_%s%s_common.mat',spike_type,method,video,addon,num2str(ThJaccard),sigma_from,baseline_std),...
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
    % end
end
end