clear;
addpath(genpath('..\evaluation'));

%% set parameters
% list_method = {'FISSA','ours'};
% list_method = {'diag_v1'};
% list_method = {'diag_v1','diag_v2'};
% list_video = {'SNR','Raw'}; % 
% alpha=10;
list_video = {'SNR'}; % {'SNR','Raw'}; 
num_video = length(list_video);
% list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'}; % ; 'Allen SDK'
list_method = {'BG subtraction'; 'FISSA'; 'CNMF'; 'AllenSDK'; 'TUnCaT'}; % ; 'Allen SDK'
num_method = length(list_method);
% std_method = 'quantile-based std comp';  % comp
% std_method = 'psd';  % comp
addon = ''; 

spike_type = '1p'; % {'include','exclude','only'};
% max_iter = 20000;
% dir_video='E:\OnePhoton videos\cropped videos\';
dir_video='..\data\1p\';
% dir_traces=dir_video;
% dir_scores='..\evaluation\1p\';
dir_traces='..\results\1p\unmixed traces\';
dir_scores='..\results\1p\evaluation\';
dir_label = [dir_video,'GT transients\'];
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};

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

load('..\template\1P_spike_tempolate.mat','filter_tempolate');
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
ii = 9;
Exp_ID = list_Exp_ID{ii};
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
        ['scores_split_CNMF_',video,'Video_p1_sumSigma',baseline_std,'.mat'],...
        ['scores_split_AllenSDK_',video,'Video_Unmix',baseline_std,'.mat'],...
        ['scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat']...
        }';
%         ['scores_',video,'Video_traces_ours_',video,'_sigma1_diag11_v1_RawSigma.mat'],...
%     list_scorefile{1}=list_scorefile{3};
    list_tracefile = {['traces_ours_',video,'',addon],...
        ['traces_FISSA_',video,''],...
        ['traces_CNMF_',video,'_p1',''],...
        ['traces_AllenSDK_',video,''],...
        ['traces_ours_',video,'',addon]...
        }';
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
            if useTF
                traces_raw_filt=conv2(raw_traces,kernel,'valid');
            else
                traces_raw_filt=raw_traces;
            end
            list_traces_raw{vid} = raw_traces;
            list_traces_raw_filt{vid} = traces_raw_filt;
            [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

%             [recall(mid,vid), precision(mid,vid), F1(mid,vid),individual_recall{mid,vid},...
%                 individual_precision{mid,vid},spikes_GT_array{mid,vid},spikes_eval_array{mid,vid}]...
%                 = GetPerformance_SpikeDetection(output,traces_unmixed_filt,thred_ratio,sigma_raw,mu_raw);
        
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
            ncells = size(output,1);
            ncells_remain = size(traces_pure,1);
            if ncells_remain < ncells
                num_miss = ncells - size(traces_pure,1);
                addpath(genpath('C:\Matlab Files\neuron_post'));
                load([dir_video,'\GT Masks merge\FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
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
                fp = fopen(sprintf('1p\\removed neurons in %s.txt',Exp_ID),'w');
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
                traces_filt = traces_sum;
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
    end
end
%%
individual_recall_array = cell2mat(individual_recall(:)');
individual_precision_array = cell2mat(individual_precision(:)');
individual_F1 = cellfun(@(x,y) 2./(1./x+1./y),individual_recall,individual_precision, 'UniformOutput',false);
individual_F1_array = 2./(1./individual_recall_array+1./individual_precision_array);

