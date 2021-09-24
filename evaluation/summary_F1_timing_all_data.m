%% ABO
clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
% %%
% dir_video='D:\ABO\20 percent 200\';
dir_video = '..\data\ABO\';
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};

list_spike_type = {'ABO'}; % {'include','exclude','only'}; % 
list_spike_type = cellfun(@(x) [x,'_BGSubs'], list_spike_type, 'UniformOutput',false);
% spike_type = 'only_MovMedianSubs_median'; % {'include','exclude','only'};
list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; 
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
addon = '_novideounmix_r2_mixout'; % '_eps=0.1'; % _downsample100 
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 

% %%
[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video));
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
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
    list_tracefile = {['traces_ours_',video,'_bgsubs'],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ,['traces_AllenSDK_',video,'']...
        }'; % _old
%         ['traces_FISSA_',video,' (tol=1e-4, max_iter=20000)'],...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(10,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_video,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_all{mid,vid} = Table_time(:,end)+Table_time_SNR;
        else
            Table_time_all{mid,vid} = squeeze(Table_time)+Table_time_SNR;
        end
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile('.\',spike_type,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        if any(mid == [2,3])
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('%s\\timing_all_methods_split %s%s x5.mat',spike_type,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');
end

%% 1p
clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
% %%
% dir_video='E:\OnePhoton videos\cropped videos\';
dir_video = '..\data\1p\';
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 

spike_type = '1p'; % {'only','include','exclude'};
list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; 
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
addon = '_novideounmix_r2_mixout'; % '_pertmin=0.16_range2_merge'; % 

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
    list_tracefile = {['traces_ours_',video,''],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ,['traces_AllenSDK_',video,''],...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(9,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_video,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_all{mid,vid} = Table_time(:,end)+Table_time_SNR;
        else
            Table_time_all{mid,vid} = squeeze(Table_time)+Table_time_SNR;
        end
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile('.\',spike_type,list_scorefile{mid}));
        list_recall_all{mid,vid} = list_recall;
        list_precision_all{mid,vid} = list_precision;
        list_F1_all{mid,vid} = list_F1;
        list_thred_ratio_all{mid,vid} = list_thred_ratio;
        if any(mid == [2,3])
            list_alpha_all{mid,vid} = list_alpha;
        end
    end
end
save(sprintf('1p\\timing_all_methods_split %s%s x5.mat',addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');


%% NAOMi
clear;
addon = '_novideounmix_r2_mixout'; % _mixout '_pertmin=0.16_eps=0.1_range'; %  
% simu_opt = '300s_10Hz_N=100_40mW_noise10+23_NA0.4,0.3'; % _NA0.4,0.3
% simu_opt = '1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c'; % 
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % 
% dir_video=['F:\NAOMi\',simu_opt,'\'];
dir_video = '..\data\ABO\';
% list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);

spike_type = 'NAOMi'; % {'include','exclude','only'};
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
    list_tracefile = {['traces_ours_',video,'_bgsubs'],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,addon_FISSA],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ['traces_AllenSDK_',video,''],...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(10,1);
    end

    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_video,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_all{mid,vid} = Table_time(:,end)+Table_time_SNR;
        else
            Table_time_all{mid,vid} = squeeze(Table_time)+Table_time_SNR;
        end
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid} = list_alpha;
        end
        
        load(fullfile('.\',spike_type,list_scorefile{mid}));
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
save(sprintf('NAOMi\\timing_%s_all_methods_split %s%s x5.mat',simu_opt,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all',...
    'list_thred_ratio_all'); % ,'list_corr_active_unmix_all','list_corr_inactive_unmix_all'


%% NAOMi vary parameter
clear;
list_variable = {'T','fs','N','power','Gaus_noise'};
struct_variable_param = struct( 'T', [30, 50, 120, 320, 1020],...
                                'fs', [3, 10, 30, 100, 300],...
                                'N', [50, 100, 200, 300, 400],...
                                'power', [20, 30, 50, 70, 100, 150],...
                                'Gaus_noise', [1, 10, 30, 50, 100]);
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
num_Exp = length(list_Exp_ID);

spike_type = 'NAOMi'; % {'include','exclude','only'};
list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; ; 'AllenSDK'
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
addon = '_novideounmix_r2_mixout'; % '_pertmin=0.16_eps=0.1_range'; %  

for vaid = 1:length(list_variable)
    variable = list_variable{vaid};
    list_param = struct_variable_param.(variable);
    num_param = length(list_param);
    [list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
        list_corr_unmix_all,list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video,num_param));
    list_N_neuron = zeros(num_Exp,num_param);

    for pid = 1:num_param
        param = list_param(pid);
    % simu_opt = sprintf('%ds_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
    % simu_opt = sprintf('120s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
    % simu_opt = sprintf('120s_30Hz_N=%d_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
    % simu_opt = sprintf('120s_30Hz_N=200_%dmW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
    if param == 1
        simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'); % 
    else
        simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23x%s_NA0.8,0.6_GCaMP6f',num2str(param)); % 
    end
    % dir_video=['F:\NAOMi\',simu_opt,'\'];
    dir_video = '..\data\NAOMi\';

    % %%
    for vid = 1:num_video
        video = list_video{vid};
        list_scorefile = {['scores_split_bgsubs_',simu_opt,'_',video,'Video_Raw_Sigma',baseline_std,'.mat'],... % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
            ['scores_split_FISSA_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
            ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_Sigma',addon,baseline_std,'.mat'],...
            ['scores_split_CNMF_',simu_opt,'_',video,'Video_p1_sumSigma',baseline_std,'.mat']...
            ['scores_split_AllenSDK_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
            }'; % ,...
    %     list_scorefile{1}=list_scorefile{3};
        list_tracefile = {['traces_ours_',video,'_bgsubs'],... % ,['traces_ours_',video,'_nooverlap'],[]
            ['traces_FISSA_',video,''],...
            ['traces_ours_',video,addon],...
            ['traces_CNMF_',video,'_p1']...
            ['traces_AllenSDK_',video,''],...
            }'; % ,...
    %     list_tracefile{1}=list_tracefile{3};
        if contains(video,'SNR')
            load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
            Table_time_SNR = Table_time';
        else
            Table_time_SNR = zeros(num_Exp,1);
        end

        for k=1:num_Exp
            load([dir_video,'GT Masks\FinalMasks_',list_Exp_ID{k},'.mat'],'FinalMasks')
            list_N_neuron(k,pid) = size(FinalMasks,3);
        end

        for mid = 1:length(list_method)
            method = list_method(mid);
            dir_FISSA = fullfile(dir_video,list_tracefile{mid});
            load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
            if mid == 1
                Table_time_all{mid,vid,pid} = Table_time(:,end)+Table_time_SNR;
            else
                Table_time_all{mid,vid,pid} = squeeze(Table_time)+Table_time_SNR;
            end
            if any(mid == [2,3])
                list_alpha_all_time{mid,vid,pid} = list_alpha;
            end

            load(fullfile('.\',spike_type,list_scorefile{mid}));
            list_recall_all{mid,vid,pid} = list_recall;
            list_precision_all{mid,vid,pid} = list_precision;
            list_F1_all{mid,vid,pid} = list_F1;
            list_thred_ratio_all{mid,vid,pid} = list_thred_ratio;
            list_corr_unmix_all{mid,vid,pid} = list_corr_unmix;
            if any(mid == [2,3])
                list_alpha_all{mid,vid,pid} = list_alpha;
            end
        end
    end
    end
    save(sprintf('NAOMi\\timing_x5_%s_GCaMP6f_all_methods_split %s%s.mat',variable,addon,baseline_std),'list_N_neuron',...
        'list_param','list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
        'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');
end

%% NAOMi vary sensor
clear;
list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
list_prot = {'GCaMP6f','GCaMP6s','jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'}; % 'jGCaMP7c','jGCaMP7b','jGCaMP7f','jGCaMP7s'
num_param = length(list_prot);
num_Exp = length(list_Exp_ID);

spike_type = 'NAOMi'; % {'include','exclude','only'};
list_method = {'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'; 'AllenSDK'}; % 'Remove overlap'; 'Percent pixels'; ; 'AllenSDK'
num_method = length(list_method);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
addon = '_novideounmix_r2_mixout'; % '_pertmin=0.16_eps=0.1_range'; %  

[list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
    list_corr_unmix_all,list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video,num_param));
list_N_neuron = zeros(num_Exp,num_param);

for pid = 1:num_param
    prot = list_prot{pid};
if contains(prot,'6')
    simu_opt = ['120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_',prot]; % 
else
    simu_opt = ['1100s_3Hz_N=200_100mW_noise10+23_NA0.8,0.6_',prot]; % 
end
% dir_video=['F:\NAOMi\',simu_opt,'\'];
dir_video = '..\data\NAOMi\';

% %%
for vid = 1:num_video
    video = list_video{vid};
    list_scorefile = {['scores_split_bgsubs_',simu_opt,'_',video,'Video_Raw_Sigma',baseline_std,'.mat'],... % ,['scores_rmoverlap_',video,'Video.mat'], ['scores_prt_',video,'Video.mat']
        ['scores_split_FISSA_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
        ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_Sigma',addon,baseline_std,'.mat'],...
        ['scores_split_CNMF_',simu_opt,'_',video,'Video_p1_sumSigma',baseline_std,'.mat']...
        ['scores_split_AllenSDK_',simu_opt,'_',video,'Video_Unmix_Sigma',baseline_std,'.mat'],...
        }'; % ,...
%     list_scorefile{1}=list_scorefile{3};
    list_tracefile = {['traces_ours_',video,'_bgsubs'],... % ,['traces_ours_',video,'_nooverlap'],[]
        ['traces_FISSA_',video,''],...
        ['traces_ours_',video,addon],...
        ['traces_CNMF_',video,'_p1']...
        ['traces_AllenSDK_',video,''],...
        }'; % ,...
%     list_tracefile{1}=list_tracefile{3};
    if contains(video,'SNR')
        load([dir_video,'SNR Video\Table_time.mat'],'Table_time');
        Table_time_SNR = Table_time';
    else
        Table_time_SNR = zeros(num_Exp,1);
    end
    
    for k=1:num_Exp
        load([dir_video,'GT Masks\FinalMasks_',list_Exp_ID{k},'.mat'],'FinalMasks')
        list_N_neuron(k,pid) = size(FinalMasks,3);
    end
    
    for mid = 1:length(list_method)
        method = list_method(mid);
        dir_FISSA = fullfile(dir_video,list_tracefile{mid});
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if mid == 1
            Table_time_all{mid,vid,pid} = Table_time(:,end)+Table_time_SNR;
        else
            Table_time_all{mid,vid,pid} = squeeze(Table_time)+Table_time_SNR;
        end
        if any(mid == [2,3])
            list_alpha_all_time{mid,vid,pid} = list_alpha;
        end
        
        load(fullfile('.\',spike_type,list_scorefile{mid}));
        list_recall_all{mid,vid,pid} = list_recall;
        list_precision_all{mid,vid,pid} = list_precision;
        list_F1_all{mid,vid,pid} = list_F1;
        list_thred_ratio_all{mid,vid,pid} = list_thred_ratio;
        list_corr_unmix_all{mid,vid,pid} = list_corr_unmix;
        if any(mid == [2,3])
            list_alpha_all{mid,vid,pid} = list_alpha;
        end
    end
end
end
save(sprintf('NAOMi\\timing_x5_sensor_GCaMP6f_all_methods_split %s%s.mat',addon,baseline_std),'list_N_neuron',...
    'list_prot','list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');

%% Merge Table_time.mat files
% dir_time = 'F:\NAOMi\300s_10Hz_N=200_40mW_noise10+23_NA0.4,0.3\traces_ours_Raw';
% part1 = load([dir_time,'\Table_time(0.01-0.05).mat']);
% part2 = load([dir_time,'\Table_time(0.1-1).mat']);
% % list_alpha = [part1.list_alpha,part2.list_alpha(:,end-3:end)];
% % Table_time = [part1.Table_time(:,1:end-1),part2.Table_time(:,end-4:end)];
% list_alpha = [part1.list_alpha,part2.list_alpha];
% Table_time = [part1.Table_time(:,1:end-1),part2.Table_time];
% save([dir_time,'\Table_time.mat'],'list_alpha','Table_time');
