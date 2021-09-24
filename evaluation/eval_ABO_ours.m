clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
% dir_video='D:\ABO\20 percent 200';
% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
dir_video = '..\data\ABO';
dir_label = [dir_video,'\GT transients'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
num_Exp=length(list_Exp_ID);

list_spike_type = {'ABO'}; % 
% list_spike_type = {'include','only','exclude'}; % 
% list_spike_type = cellfun(@(x) [x,'_noBGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'}; 
list_baseline_std = {'_ksd-psd'}; % , ''
% baseline_std = ''; % '_psd'; % '_ksd-psd'; % 
MovMedianSubs = contains(list_spike_type{1},'MovMedian'); % false;

method = 'ours'; % {'FISSA','ours'}
list_video={'Raw','SNR'}; % {'Raw','SNR'}
addon = '_novideounmix_r2_mixout'; %,_fixed_alpha '_eps=0.1'; % _n_iter _fixed_alpha
list_part1={''}; % , '_pertmin=0.5', '_pertmin=0.16'
% part1 = ''; %, '_diag11'
list_part2 = {''}; % , '_eps=0.1'
% part2=''; % ,'v2'
list_part3 = {''}; % , '_range'
max_alpha = inf;

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
% std_method = 'quantile-based std comp';  % comp
% std_method = 'psd';  % comp
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            for i1 = 1:length(list_part1)
                part1 = list_part1{i1};
            for i2 = 1:length(list_part2)
                part2 = list_part2{i2};
            for i3 = 1:length(list_part3)
                part3 = list_part3{i3};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    list_thred_ratio=30:5:80; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                else
                    list_thred_ratio=30:5:80; % GCaMP6f; % 0.3:0.3:3; % 1:0.5:6; % 
                end
            else
                list_thred_ratio=6:0.5:9; % 6:12; % 9:16; % 
            end
            folder = sprintf('traces_%s_%s%s%s%s%s',method,video,part1,part2,part3,addon);
            dir_FISSA = fullfile(dir_video,folder);
            useTF = strcmp(video, 'Raw');

%             list_alpha = [0.1, 0.2, 0.3, 0.5, 1]; %
%             list_alpha = 1*[0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]; %
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

            num_alpha=length(list_alpha);
            num_ratio=length(list_thred_ratio);
            [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_alpha, num_ratio));

            if useTF
%                 dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
                load('..\template\GCaMP6f_spike_tempolate_mean.mat','filter_tempolate');
                dFF = filter_tempolate;
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

                load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
                traces_raw = traces'-bgtraces';
                if MovMedianSubs
                    traces_raw = traces_raw - movmedian(traces_raw,900,2);
                end
                if useTF
                    traces_raw_filt=conv2(traces_raw,kernel,'valid');
                else
                    traces_raw_filt=traces_raw;
                end
                [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

                fprintf('Neuron Toolbox');
                for jj = 1:num_alpha % 
                    alpha = list_alpha(jj);
                    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\balpha=%6.3f: ',alpha);
                    load(fullfile(dir_FISSA,sprintf('alpha=%6.3f',alpha),[Exp_ID,'.mat']),'traces_nmfdemix');
                    if MovMedianSubs
                        traces_nmfdemix = traces_nmfdemix - movmedian(traces_nmfdemix,900,2);
                    end
                    if useTF
                        traces_unmixed_filt=conv2(traces_nmfdemix',kernel,'valid');
                    else
                        traces_unmixed_filt=traces_nmfdemix';
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
                    if alpha >= 100
                        fprintf('\b');
                    end
                    if alpha >= 1000
                        fprintf('\b');
                    end
                end
                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
            end
            fprintf('\b\b\b\b\b\b\b\b\b\b\b');

            %%
            if ~exist(spike_type)
                mkdir(spike_type);
            end
            save(sprintf('%s\\scores_split_%s_%sVideo%s%s%s%s_%sSigma%s.mat',spike_type,method,video,part1,part2,part3,addon,sigma_from,baseline_std),...
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
    end
end
end