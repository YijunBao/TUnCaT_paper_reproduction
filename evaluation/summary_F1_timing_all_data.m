%% ABO
clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
% %%
spike_type = 'ABO'; 
% dir_video='D:\ABO\20 percent 200\';
dir_video = ['..\data\',spike_type,'\'];
% dir_traces=dir_video;
% dir_scores=['..\evaluation\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};

list_spike_type = {'ABO'}; % {'include','exclude','only'}; % 
% list_spike_type = cellfun(@(x) [x,'_BGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'only_MovMedianSubs_median'; % {'include','exclude','only'};
list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; 
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
addon = ''; % '_eps=0.1'; % _downsample100 
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 

% %%
[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video));
for tid = 1:length(list_spike_type)
%     spike_type = list_spike_type{tid}; % 
for vid = 1:num_video
    video = list_video{vid};
%     list_scorefile = {['scores_split_bgsubs_',video,'Video',baseline_std,'.mat'],... 
    list_scorefile = {['scores_split_bgsubs_',video,'Video',baseline_std,'.mat'],... 
        ['scores_split_FISSA_',video,'Video_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_CNMF_',video,'Video_p1_sum',baseline_std,'.mat']...
        ,['scores_split_AllenSDK_',video,'Video_Unmix',baseline_std,'.mat']...
        }'; % _old
%         ['scores_FISSA_',video,'Video_UnmixSigma',baseline_std,' (tol=1e-4, max_iter=20000).mat'],...
        % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
%         ['scores_',video,'Video_traces_ours_',video,'_sigma1_diag11_v1_RawSigma',baseline_std,'.mat'],...
%     list_scorefile{1}=list_scorefile{3};
%     list_tracefile = {['traces_ours_',video],... % ,['traces_ours_',video,'_nooverlap'],[]
    list_tracefile = {['traces_ours_',video,''],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ,['traces_AllenSDK_',video,'']...
        }'; % _old
%         ['traces_FISSA_',video,' (tol=1e-4, max_iter=20000)'],...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_traces,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(10,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_traces,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_temp = Table_time(:,end);
        elseif size(Table_time,1) == 1
            Table_time_temp = Table_time';
        else
            Table_time_temp = Table_time;
        end
        Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
        Table_time_all{mid,vid} = Table_time_temp;
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile(dir_scores,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        if any(mid == [2,3])
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('%s\\timing_all_methods_split %s%s x5.mat',dir_scores,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');
end


%% 1p
clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
% %%
% dir_video='E:\OnePhoton videos\cropped videos\';
spike_type = '1p'; % dir_video='D:\ABO\20 percent 200\';
dir_video = ['..\data\',spike_type,'\'];
% dir_traces=dir_video;
% dir_scores=['..\evaluation\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 

list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; 
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
addon = ''; % '_pertmin=0.16_range2_merge'; % 

% %%
[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video));
for vid = 1:num_video
    video = list_video{vid};
    list_scorefile = {['scores_split_bgsubs_',video,'Video',baseline_std,'.mat'],... % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
        ['scores_split_FISSA_',video,'Video_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_CNMF_',video,'Video_p1_sumSigma',baseline_std,'.mat']...
        ,['scores_split_AllenSDK_',video,'Video_Unmix',baseline_std,'.mat']...
        }'; % ,...
%     list_scorefile{1}=list_scorefile{3};
    list_tracefile = {['traces_ours_',video,addon],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ,['traces_AllenSDK_',video,''],...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_traces,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(9,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_traces,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_temp = Table_time(:,end);
        elseif size(Table_time,1) == 1
            Table_time_temp = Table_time';
        else
            Table_time_temp = Table_time;
        end
        Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
        Table_time_all{mid,vid} = Table_time_temp;
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile(dir_scores,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        if any(mid == [2,3])
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('%s\\timing_all_methods_split %s%s x5.mat',dir_scores,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');


%% NAOMi
clear;
addon = ''; % _mixout '_pertmin=0.16_eps=0.1_range'; %  
% simu_opt = '300s_10Hz_N=100_40mW_noise10+23_NA0.4,0.3'; % _NA0.4,0.3
% simu_opt = '1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c'; % 
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % 
% dir_video=['F:\NAOMi\',simu_opt,'\'];
spike_type = 'NAOMi'; % dir_video='D:\ABO\20 percent 200\';
dir_video = ['..\data\',spike_type,'\'];
% dir_traces=dir_video;
% dir_scores=['..\evaluation\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];
% list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);

list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; ; 'AllenSDK'
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
baseline_std = '_ksd-psd'; % '_psd'; % ''; % _corr_active

% %%
[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_corr_unmix_all,list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video));
for vid = 1:num_video
    video = list_video{vid};
%     if contains(video,'SNR')
%         addon_FISSA = '_corr';
%     else
        addon_FISSA = '';
%     end
    list_scorefile = {['scores_split_bgsubs_',simu_opt,'_',video,'Video_Raw_Sigma',baseline_std,'.mat'],... % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
        ['scores_split_FISSA_',simu_opt,'_',video,'Video_Unmix_Sigma',addon_FISSA,baseline_std,'.mat'],...
        ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_Sigma',addon,baseline_std,'.mat'],...
        ['scores_split_CNMF_',simu_opt,'_',video,'Video_p1_sumSigma',baseline_std,'.mat']...
        ['scores_split_AllenSDK_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
        }'; % ,...
%     list_scorefile{1}=list_scorefile{3};
    list_tracefile = {['traces_ours_',video,addon],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,addon_FISSA],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ['traces_AllenSDK_',video,''],...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_traces,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(10,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_traces,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_temp = Table_time(:,end);
        elseif size(Table_time,1) == 1
            Table_time_temp = Table_time';
        else
            Table_time_temp = Table_time;
        end
        Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
        Table_time_all{mid,vid} = Table_time_temp;
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile(dir_scores,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        list_corr_unmix_all{mid,vid} = list_corr_unmix;
%         list_corr_active_unmix_all{mid,vid} = list_corr_unmix_active;
%         list_corr_inactive_unmix_all{mid,vid} = list_corr_unmix_inactive;
        if any(mid == [2,3])
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('%s\\timing_%s_all_methods_split %s%s x5.mat',dir_scores,simu_opt,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all',...
    'list_thred_ratio_all'); % ,'list_corr_active_unmix_all','list_corr_inactive_unmix_all'


%% ABO CaImAn vs SUNS+TUnCaT
spike_type = 'ABO'; % dir_video='D:\ABO\20 percent 200\';
dir_video = ['..\data\',spike_type,'\'];
% dir_traces=dir_video;
% dir_scores=['..\evaluation\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};
% list_Exp_ID = list_Exp_ID([2,3]);
SUNS = 'SUNS_complete'; % 'SUNS_noSF'; % 
folder_SUNS = [SUNS,'\output_masks\'];

list_spike_type = {'ABO'}; % {'include','exclude','only'}; % 
% list_spike_type = cellfun(@(x) [x,'_BGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'only_MovMedianSubs_median'; % {'include','exclude','only'};
list_method = {'TUnCaT'; 'SUNS+TUnCaT'; 'SUNS+CNMF'; 'CaImAn'}; % 'Remove overlap'; 'Percent pixels'; 
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
addon = '';
addon1 = ''; % '_eps=0.1'; % _downsample100 
addon2 = ''; % '_eps=0.1'; % _downsample100 
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
list_TP = {'_hasFNFP','_common'}; % '_onlyTP'; % 

% %%
[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video));
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
for TPid = 1:length(list_TP)
    TP = list_TP{TPid}; % 
for vid = 1:num_video
    video = list_video{vid};
%     list_scorefile = {['scores_split_bgsubs_',video,'Video',baseline_std,'.mat'],... 
    list_scorefile = {['scores_split_ours_',video,'Video',addon1,'_UnmixSigma',baseline_std,'.mat'],...
        ['scores_split_',SUNS,'+ours_',video,'Video',addon2,'_0.5_UnmixSigma',baseline_std,TP,'.mat'],...
        ['scores_split_',SUNS,'+CNMF_',video,'Video_p1_0.5_sum',baseline_std,TP,'.mat']...
        ['scores_split_CaImAn_',video,'Video_0.5_sum',baseline_std,TP,'.mat']...
        }'; % hasFNFP
%     list_scorefile = {['scores_split_ours_',video,'Video',addon,'_0.5_UnmixSigma',baseline_std,'_onlyTP.mat'],...
%         ['scores_split_CNMF_',video,'Video_p1_0.5_sum',baseline_std,'_onlyTP.mat']...
%         ['scores_split_CaImAn_',video,'Video_0.5_sum',baseline_std,'_onlyTP.mat']...
%         }'; % onlyTP
%     list_tracefile = {['traces_ours_',video],... % ,['traces_ours_',video,'_nooverlap'],[]
    list_tracefile = {['traces_ours_',video,addon1],...
        ['traces_',SUNS,'+ours_',video,addon2],...
        ['traces_',SUNS,'+CNMF_',video,'_p1']...
        ['CaImAn-Batch_',video]...
        }'; % ,'\275'

    for mid = 1:num_method
        if contains(list_tracefile{mid},'SUNS')
            load([dir_traces,folder_SUNS,'Output_Info_All.mat'],'list_time')
            Table_time_SNR = list_time(:,end);
        elseif contains(video,'SNR')
            load([dir_traces,'SNR Video\Table_time.mat'],'Table_time');
            Table_time_SNR = Table_time';
        else
            Table_time_SNR = zeros(10,1);
        end
    
        method = list_method(mid);
        dir_FISSA = fullfile(dir_traces,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if size(Table_time,1) == 1
            Table_time_temp = Table_time';
        else
            Table_time_temp = Table_time;
        end
        Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
        Table_time_all{mid,vid} = Table_time_temp;
        if any(mid == [1,2]) % mid == 1 % 
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile(dir_scores,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        if any(mid == [1,2]) % mid == 1 % 
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('%s\\timing_all_methods_split %s%s %s%s.mat',dir_scores,addon,baseline_std,SUNS,TP),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');
end
end


