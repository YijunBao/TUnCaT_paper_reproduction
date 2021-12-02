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
addon = ''; % '_pertmin=0.16_eps=0.1_range'; %  

for vaid = 1:length(list_variable)
    variable = list_variable{vaid};
    list_param = struct_variable_param.(variable);
    num_param = length(list_param);
    [list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
        list_corr_unmix_all,list_alpha_all_time,Table_time_all] = deal(cell(num_method,num_video,num_param));
    list_N_neuron = zeros(num_Exp,num_param);

    for pid = 1:num_param
        param = list_param(pid);
        switch variable
            case 'T'
                simu_opt = sprintf('%ds_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'fs'
                simu_opt = sprintf('120s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'N'
                simu_opt = sprintf('120s_30Hz_N=%d_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'power'
                simu_opt = sprintf('120s_30Hz_N=200_%dmW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'Gaus_noise'
                if param ~= 1
                    simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23x%s_NA0.8,0.6_GCaMP6f',num2str(param)); % 
                else
                    simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'); % 
                end
        end
    % dir_video=['F:\NAOMi\',simu_opt,'\'];
    dir_video = ['..\data\',spike_type,'\'];
    dir_traces=['..\results\',spike_type,'\unmixed traces\'];
    dir_scores=['..\results\',spike_type,'\evaluation\'];

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
        list_tracefile = {['traces_ours_',video,addon],... % ,['traces_ours_',video,'_nooverlap'],[]
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
            Table_time_all{mid,vid,pid} = Table_time_temp;
            if any(mid == [2,3])
                list_alpha_all_time{mid,vid,pid} = list_alpha;
            end

            load(fullfile(dir_scores,list_scorefile{mid}));
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
    save(sprintf('%s\\timing_x5_%s_GCaMP6f_all_methods_split %s%s.mat',dir_scores,variable,addon,baseline_std),'list_N_neuron',...
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
addon = ''; % '_pertmin=0.16_eps=0.1_range'; %  

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
dir_video = ['..\data\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];

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
    list_tracefile = {['traces_ours_',video,addon],... % ,['traces_ours_',video,'_nooverlap'],[]
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
        Table_time_all{mid,vid,pid} = Table_time_temp;
    if any(mid == [2,3])
            list_alpha_all_time{mid,vid,pid} = list_alpha;
        end
        
        load(fullfile(dir_scores,list_scorefile{mid}));
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
save(sprintf('%s\\timing_x5_sensor_GCaMP6f_all_methods_split %s%s.mat',dir_scores,addon,baseline_std),'list_N_neuron',...
    'list_prot','list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');


%% NAOMi vary parameter compared to alpha=1
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
list_addon={'_alpha=1','_novideounmix_r2_mixout'}; % 
num_addon = length(list_addon);
list_video= {'Raw','SNR'}; % 'Raw','SNR'
num_video = length(list_video);
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
% addon = ''; % '_pertmin=0.16_eps=0.1_range'; %  

for vaid = 1:length(list_variable)
    variable = list_variable{vaid};
    list_param = struct_variable_param.(variable);
    num_param = length(list_param);
    [list_recall_all,list_precision_all,list_F1_all,list_alpha_all,list_thred_ratio_all,...
        list_corr_unmix_all,list_alpha_all_time,Table_time_all] = deal(cell(num_addon,num_video,num_param));
    list_N_neuron = zeros(num_Exp,num_param);

    for pid = 1:num_param
        param = list_param(pid);
        switch variable
            case 'T'
                simu_opt = sprintf('%ds_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'fs'
                simu_opt = sprintf('120s_%dHz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'N'
                simu_opt = sprintf('120s_30Hz_N=%d_100mW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'power'
                simu_opt = sprintf('120s_30Hz_N=200_%dmW_noise10+23_NA0.8,0.6_GCaMP6f',param); % 
            case 'Gaus_noise'
                if param ~= 1
                    simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23x%s_NA0.8,0.6_GCaMP6f',num2str(param)); % 
                else
                    simu_opt = sprintf('120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'); % 
                end
        end
    % dir_video=['F:\NAOMi\',simu_opt,'\'];
    dir_video = ['..\data\',spike_type,'\'];
    dir_traces=['..\results\',spike_type,'\unmixed traces\'];
    dir_scores=['..\results\',spike_type,'\evaluation\'];

    % %%
    for vid = 1:num_video
        video = list_video{vid};
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

        for aid = 1:num_addon
            addon = list_addon{aid};
            if contains(addon,'alpha=')
                scorefile = ['scores',addon,'_ours_',simu_opt,'_',video,'Video_Unmix_compSigma','_novideounmix_r2_mixout',baseline_std,'.mat']; % 
                tracefile = ['traces_ours_',video,'_novideounmix_r2_mixout']; % ,...
            else
                scorefile = ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_compSigma',addon,baseline_std,'.mat']; % 
                tracefile = ['traces_ours_',video,addon]; % ,...
            end
            dir_FISSA = fullfile(dir_traces,tracefile);
            load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
            if contains(addon,'alpha=')
                temp = split(addon,'=');
                alpha = str2double(temp{end});
                ind_alpha = find(list_alpha == alpha);
                Table_time = Table_time(:,[ind_alpha,end]);
                list_alpha = alpha;
                list_final_alpha_all = alpha;
            end

            Table_time_temp = Table_time;
            Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
            Table_time_all{aid,vid,pid} = Table_time_temp;
            list_alpha_all_time{aid,vid,pid} = list_alpha;
            
            load(fullfile(dir_scores,scorefile));
            list_recall_all{aid,vid,pid} = list_recall;
            list_precision_all{aid,vid,pid} = list_precision;
            list_F1_all{aid,vid,pid} = list_F1;
            list_thred_ratio_all{aid,vid,pid} = list_thred_ratio;
            list_corr_unmix_all{aid,vid,pid} = list_corr_unmix;
            list_alpha_all{aid,vid,pid} = list_alpha;
            final_alpha_all{aid,vid,pid} = list_final_alpha_all;
        end
    end
    end
    save(sprintf('%s\\timing_%s_GCaMP6f_opt_alpha_1 %s%s.mat',dir_scores,variable,addon,baseline_std),'list_N_neuron',...
        'list_param','list_alpha_all_time','Table_time_all','list_method','list_video','list_corr_unmix_all',...
        'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all');
end

%% NAOMi vary sensor compared to alpha=1
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
addon = ''; % '_pertmin=0.16_eps=0.1_range'; %  

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
dir_video = ['..\data\',spike_type,'\'];
dir_traces=['..\results\',spike_type,'\unmixed traces\'];
dir_scores=['..\results\',spike_type,'\evaluation\'];

% %%
for vid = 1:num_video
    video = list_video{vid};
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
    
    for aid = 1:num_addon
        addon = list_addon{aid};
        if contains(addon,'alpha=')
            scorefile = ['scores',addon,'_ours_',simu_opt,'_',video,'Video_Unmix_compSigma','_novideounmix_r2_mixout',baseline_std,'.mat']; % 
            tracefile = ['traces_ours_',video,'_novideounmix_r2_mixout']; % ,...
        else
            scorefile = ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_compSigma',addon,baseline_std,'.mat']; % 
            tracefile = ['traces_ours_',video,addon]; % ,...
        end
        dir_FISSA = fullfile(dir_video,tracefile);
        load(fullfile(dir_FISSA,'Table_time.mat')); % ,'Table_time', 'list_alpha'
        if contains(addon,'alpha=')
            temp = split(addon,'=');
            alpha = str2double(temp{end});
            ind_alpha = find(list_alpha == alpha);
            Table_time = Table_time(:,[ind_alpha,end]);
            list_alpha = alpha;
            list_final_alpha_all = alpha;
        end
        Table_time_temp = Table_time;
        Table_time_temp(:,end) = Table_time_temp(:,end)+Table_time_SNR;
        Table_time_all{aid,vid,pid} = Table_time_temp;
        list_alpha_all_time{aid,vid,pid} = list_alpha;
        
        load(fullfile('.\',spike_type,scorefile));
        list_recall_all{aid,vid,pid} = list_recall;
        list_precision_all{aid,vid,pid} = list_precision;
        list_F1_all{aid,vid,pid} = list_F1;
        list_thred_ratio_all{aid,vid,pid} = list_thred_ratio;
        list_corr_unmix_all{aid,vid,pid} = list_corr_unmix;
        list_alpha_all{aid,vid,pid} = list_alpha;
        final_alpha_all{aid,vid,pid} = list_final_alpha_all;
    end
end
end
save(sprintf('%s\\timing_sensor_GCaMP6f_opt_alpha_1 %s%s.mat',dir_scores,addon,baseline_std),'list_N_neuron',...
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
