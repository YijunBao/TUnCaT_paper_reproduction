clear;
addpath(genpath('..\evaluation'));

%% set parameters
% list_method = {'FISSA','ours'};
% list_method = {'diag_v1'};
% list_method = {'diag_v1','diag_v2'};
% list_video = {'SNR','Raw'}; % 
% alpha=10;
list_video = {'Raw'};  % 'SNR',
num_video = length(list_video);
% list_method = {'Clean trace'; 'Background subtraction'; 'Percent pixels'; 'FISSA'; 'Our unmixing'; 'CNMF'};
% list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'}; % 'Remove overlap'; 'Percent pixels'; ; 'AllenSDK'
list_method = {'BG subtraction'; 'FISSA'; 'CNMF'; 'AllenSDK'; 'TUnCaT'}; % ; 'Allen SDK'
num_method = length(list_method);
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

spike_type = 'NAOMi'; % {'include','exclude','only'};
% fs = 30;
% simu_opt = '100s_30Hz_N=400_40mW_noise10+23'; % _NA0.4,0.3
% simu_opt = '1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c'; % _NA0.4,0.3
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % _NA0.4,0.3
addon = '';
simu_opt_split = split(simu_opt,'_');
fs = str2double(simu_opt_split{2}(1:end-2));
% dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
dir_video='..\data\NAOMi\';
% dir_traces=dir_video;
% dir_scores='..\evaluation\NAOMi\';
dir_traces='..\results\NAOMi\unmixed traces\';
dir_scores='..\results\NAOMi\evaluation\';
dir_label = [dir_video,'GT transients\'];
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
num_Exp=length(list_Exp_ID);

load(['..\template\filter_template 100Hz ',simu_opt_split{end},'_ind_con=10.mat'],'template');
fs_template = 100;
% load('filter_template 30Hz jGCaMP7s.mat','template');
% fs_template = 30;
[val_max, loc_max] = max(template);
peak_time = loc_max/fs_template;
loc_e = find(template>max(template)*exp(-1),1,'last');
decay = (loc_e - loc_max)/fs_template;
lag = round(peak_time*fs);

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
dFF = Poisson_filt/sum(Poisson_filt);
kernel=fliplr(dFF);
% lag = ceil(fs*rise) + ceil(fs*rise*3); % 3;
cons = ceil(fs*decay*0.1);
% lag = round(length(kernel)/2);
            
[recall,precision,F1] = deal(zeros(num_method,num_video));
[individual_recall,individual_precision,individual_correlation,spikes_GT_array,spikes_eval_array] = deal(cell(num_method,num_video));
[list_traces_clean, list_traces_raw_filt, list_spike_frames] = deal(cell(1,num_video));
[list_traces_unmixed,list_traces_unmixed_filt] = deal(cell(num_method,num_video));

%% choose video file and and load manual spikes
ii = 8; % 7;
Exp_ID = list_Exp_ID{ii};

%%
for vid = 1:length(list_video)
    video = list_video{vid};
    useTF = contains(video, 'Raw');
%     if useTF
%         kernel=fliplr(dFF);
%         lag = [-peak_time/2, fs*decay/2-peak_time];
%     else
%         kernel = 1;
%         lag = 1+[peak_time/2, fs*decay/2];
%     end
%     sigma_from = 'Raw';
    list_scorefile = {['scores_split_bgsubs_',simu_opt,'_',video,'Video_Raw_Sigma',baseline_std,'.mat'],... % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
        ['scores_split_FISSA_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
        ['scores_split_CNMF_',simu_opt,'_',video,'Video_p1_sumSigma',baseline_std,'.mat'],...
        ['scores_split_AllenSDK_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
        ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_Sigma',addon,baseline_std,'.mat']...
        }'; % ,...
%     list_scorefile{1}=list_scorefile{5};
    list_tracefile = {['traces_ours_',video,'',addon],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_CNMF_',video,'_p1'],...
        ['traces_AllenSDK_',video],...
        ['traces_ours_',video,addon]...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};

    load([dir_video,'\GT Masks\Traces_etc_',Exp_ID,'.mat'],'spikes','clean_traces','Masks')
    load([dir_video,'\GT Masks\FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
    folder = list_tracefile{1};
%     dir_FISSA = fullfile(dir_video,[folder(1:end-3),'SNR']);
    dir_FISSA = fullfile(dir_video,replace(folder,video,'SNR'));
    load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces');
    if useTF
        length_kernel_py = size(clean_traces,2) - size(traces,1)+1;
        length_diff = length_kernel_py - length(dFF);
        if length_diff > 0
            kernel = padarray(kernel,[0,length_diff],'replicate','pre');
            dFF = fliplr(kernel);
        elseif length_diff < 0
            kernel = kernel(1-length_diff:end);
            dFF = fliplr(kernel);
        end
    end
    ncells = size(clean_traces,1);
%     calcium = clean_traces;
    if useTF
        calcium = clean_traces;
    else
        calcium = conv2(clean_traces,kernel,'valid');
    end
    list_traces_clean{vid} = calcium;
    spikes_cell = mat2cell(spikes,ones(ncells,1),size(spikes,2));
    spikes_frames = cellfun(@(x) find(x)*fs/fs_template, spikes_cell, 'UniformOutput', false);
    hasGT = find(~cellfun(@isempty, spikes_frames));
    if useTF
        list_spike_frames{vid} = cellfun(@(x) x-peak_time*fs,spikes_frames, 'UniformOutput', false);
        calcium_filt = conv2(calcium,kernel,'valid');
    else
        list_spike_frames{vid} = spikes_frames;
        calcium_filt = calcium;
    end
    [output, spikes_GT_line] = GT_transient_NAOMi_split(calcium_filt,spikes_frames);
%     if useTF
%         calcium = conv2(calcium,kernel,'valid');
%     end

    for mid = 1:length(list_method)
        method = list_method(mid);
        %% find optimal alpha and thred_ratio from saved results
        load(fullfile(dir_scores,list_scorefile{mid}),'list_recall',...
            'list_precision','list_F1','list_alpha','list_thred_ratio');
        folder = list_tracefile{mid};
        dir_FISSA = fullfile(dir_traces,folder);
%         array_F1 = squeeze(list_F1(ii,:,:));

        %% Load the raw traces
        if mid==1 % bgsubs
            array_F1 = squeeze(list_F1(ii,:));
            if size(array_F1,2)~=1
                array_F1=array_F1';
            end
            [F1_max,ind] = max(array_F1(:));
            [ind_thred_ratio,aid] = ind2sub(size(array_F1),ind);
            thred_ratio = list_thred_ratio(ind_thred_ratio);

            load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
            raw_traces=traces'-bgtraces';
            if useTF
                traces_raw_filt=conv2(raw_traces,kernel,'valid');
            else
                traces_raw_filt=raw_traces;
            end
            list_traces_raw_filt{vid} = traces_raw_filt;
            list_traces_unmixed{mid,vid} = raw_traces;
            list_traces_unmixed_filt{mid,vid} = traces_raw_filt;
            traces_unmixed_filt = traces_raw_filt;
            [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
            mu = mu_raw;
            sigma = sigma_raw;
        
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
                if di ~= num_miss
                    error('Need manual selection');
                end
                fp = fopen(sprintf('NAOMi\\removed neurons in %s Exp %s.txt',simu_opt,Exp_ID),'w');
                fprintf(fp, 'Neuron %d is missing',ind_remove); 
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
%             traces_sum = C_gt+YrA_gt;
%             if useTF
%                 traces_pure=conv2(traces_pure,kernel,'valid');
%                 noise_pure=conv2(noise_pure,kernel,'valid');
%                 traces_sum=conv2(traces_sum,kernel,'valid');
%             end
            [~, sigma_pure] = SNR_normalization(noise_pure,std_method,baseline_method);
            [mu_pure, ~] = SNR_normalization(traces_pure,std_method,baseline_method);
            [mu_sum, sigma_sum] = SNR_normalization(traces_sum,std_method,baseline_method);
%             if strcmp(sigma_from,'pure')
%                 sigma = sigma_pure;
%                 mu = mu_pure;
%                 traces_filt = traces_pure;
%             else
                sigma = sigma_sum;
                mu = mu_sum;
                traces_unmixed_filt = traces_sum;
%             end
            if useTF
                traces_unmixed_filt=conv2(traces_unmixed_filt,kernel,'valid');
            end
            list_traces_unmixed{mid,vid} = traces_sum;
            list_traces_unmixed_filt{mid,vid} = traces_unmixed_filt;

%             recall(mid,vid)=list_recall(ind_thred_ratio,aid);
%             precision(mid,vid)=list_precision(ind_thred_ratio,aid);
%             F1(mid,vid)=list_F1(ind_thred_ratio,aid);
%             individual_recall{mid,vid}=saved_result.individual_recall{ii,ind_thred_ratio};
%             individual_precision{mid,vid}=saved_result.individual_precision{ii,ind_thred_ratio};
%             spikes_GT_array{mid,vid}=saved_result.spikes_GT_array{ii,ind_thred_ratio};
%             spikes_eval_array{mid,vid}=saved_result.spikes_eval_array{ii,ind_thred_ratio};
        
        else % if mid==2,3
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
            
            if useTF
                traces_unmixed_filt=conv2(unmixed_traces,kernel,'valid');
            else
                traces_unmixed_filt=unmixed_traces;
            end
            list_traces_unmixed{mid,vid} = unmixed_traces;
            list_traces_unmixed_filt{mid,vid} = traces_unmixed_filt;
            [mu_unmixed, sigma_unmixed] = SNR_normalization(traces_unmixed_filt,std_method,baseline_method);

%             if mid==1
%                 sigma = sigma_raw;
%             else % mid==2,4
                mu = mu_unmixed;
                sigma = sigma_unmixed;
%             end
        end

%         [output, spikes_GT_line] = GT_transient_NAOMi_split(calcium_filt,spikes_frames);
        [recall(mid,vid), precision(mid,vid), F1(mid,vid),individual_recall{mid,vid},...
            individual_precision{mid,vid},spikes_GT_array{mid,vid},spikes_eval_array{mid,vid}]...
            = GetPerformance_SpikeDetection_simulation_trace_split(...
            output,traces_unmixed_filt,thred_ratio,sigma,mu);
%                 = GetPerformance_SpikeDetection_simulation_last(...
%                 spikes_frames,lag,traces_raw_filt,thred_ratio,sigma_raw,mu_raw,cons,fs,decay);
%                 = GetPerformance_SpikeDetection(output,traces_unmixed_filt,thred_ratio,sigma_raw,mu_raw);
    end
end
%%
individual_recall_array = cell2mat(individual_recall(:)');
individual_precision_array = cell2mat(individual_precision(:)');
individual_F1 = cellfun(@(x,y) 2./(1./x+1./y),individual_recall,individual_precision, 'UniformOutput',false);
individual_F1_array = 2./(1./individual_recall_array+1./individual_precision_array);

%% Plot masks and index
% addpath(genpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools'));
% plot_masks_id(FinalMasks,Masks);

%% correlation
% for vid = 1:length(list_video)
%     clean_traces = list_traces_clean{vid};
%     ncells = size(clean_traces,1);
%     for mid = 1:length(list_method)
%         traces_unmixed = list_traces_unmixed{mid,vid};
%         list_correlation = zeros(ncells,1);
%         for n = 1:ncells
%             list_correlation(n) = corr(traces_unmixed(n,:)',clean_traces(n,:)');
%         end
%         individual_correlation{mid,vid} = list_correlation;
%     end
% end
% individual_correlation_array = cell2mat(individual_correlation(:)');
