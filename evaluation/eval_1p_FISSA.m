clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));

%% choose video file
% dir_video='E:\OnePhoton videos\cropped videos\';
% dir_label = [dir_video,'split\'];
dir_video = '..\data\ABO';
dir_label = [dir_video,'\GT transients'];
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% list_Exp_ID = list_Exp_ID(1:5);
% spike_type = 'exclude'; % {'include','exclude','only'};
list_spike_type = {'1p'}; % {'only','include','exclude'};
% exclude_alone = false;
% video='Raw'; % {'Raw','SNR'}
list_video={'Raw','SNR'};
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'};
addon = ''; % '_eps=0.1'; % 

list_baseline_std = {'_ksd-psd'}; % '', 
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
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
    for vid = 1:length(list_video)
        video = list_video{vid};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    list_thred_ratio=100:50:600;% jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                else
                    list_thred_ratio=00:100:800; % jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                end
            else
                list_thred_ratio=0:2:20; % 4:12; % 6:0.5:9; % 8:2:20; % 
            end                
        folder = sprintf('traces_FISSA_%s%s',video,addon);
        dir_FISSA = fullfile(dir_video,folder);
        useTF = strcmp(video, 'Raw');

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

        num_Exp=length(list_Exp_ID);
        num_alpha=length(list_alpha);
        num_ratio=length(list_thred_ratio);
        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_alpha, num_ratio));

        if useTF
%             dFF = h5read('E:\OnePhoton videos\1P_spike_tempolate.h5','/filter_tempolate')';
            load('..\template\1P_spike_tempolate.mat','filter_tempolate');
            dFF = filter_tempolate;
            dFF = dFF(dFF>exp(-1));
            dFF = dFF'/sum(dFF);
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

            load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'raw_traces');
            if useTF
                traces_raw_filt=conv2(raw_traces,kernel,'valid');
            else
                traces_raw_filt=raw_traces;
            end
            [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

            fprintf('Neuro Toolbox');
            for jj = 1:num_alpha
                alpha = list_alpha(jj);
                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\balpha=%6.3f: ',alpha);
                load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'unmixed_traces');
                if useTF
                    traces_unmixed_filt=conv2(unmixed_traces,kernel,'valid');
                else
                    traces_unmixed_filt=unmixed_traces;
                end
                [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

        %         fprintf('Neuron Toolbox');
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
                        = GetPerformance_SpikeDetection_split(output,traces_unmixed_filt,thred_ratio,sigma,mu_unmixed);
                    list_recall(ii,jj,kk)=recall; 
                    list_precision(ii,jj,kk)=precision;
                    list_F1(ii,jj,kk)=F1;
                end
        %         fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
            end
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

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
        save(sprintf('%s\\scores_split_FISSA_%sVideo_%sSigma%s%s.mat',...
            spike_type,video,sigma_from,addon,baseline_std),...
            'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
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
    end
    end
end
end