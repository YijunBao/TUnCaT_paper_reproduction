% clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
list_spike_type = {'ABO','NAOMi','1p'};
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % 

for ind = 1:length(list_spike_type)
    spike_type = list_spike_type{ind};
    dir_video = ['..\data\',spike_type];
%     dir_traces=dir_video;
%     dir_scores=['..\evaluation\',spike_type,'\'];
    dir_traces=['..\results\',spike_type,'\unmixed traces\'];
    dir_scores=['..\results\',spike_type,'\evaluation\'];
    
    %% common
    addon = ''; % '_eps=0.1'; % 
    addon_nobin = '';
    list_nbin = [1,2,4,8,16,32,64,100]; % 32 % 
    list_bin_option = {'downsample'}; % 'sum','mean',
    list_video= {'Raw','SNR'}; % 'Raw'
    num_nbin = length(list_nbin);
    % variable = 'bin_option';
    baseline_std = '_ksd-psd'; % '', 
    num_video = length(list_video);
    method = 'ours'; % {'FISSA','ours'}
    num_bin_option = length(list_bin_option);
    part1='';
    part2='';
    part3='';
    sigma_from='Unmix';

    % %%
    [list_recall_all,list_precision_all,list_F1_all,list_thred_ratio_all,...
        list_alpha_all,list_alpha_all_time,Table_time_all,list_corr_unmix_all] ...
        = deal(cell(num_nbin,num_bin_option,num_video));
    for vid = 1:num_video
        video = list_video{vid};
        for oid = 1:num_bin_option
            bin_option = list_bin_option{oid};
    %         ind = (vid-1)*num_bin_option+oid;
            for bid = 1:num_nbin
                nbin = list_nbin(bid);
        %         folder = sprintf('traces_ours_%s (tol=1e-4, max_iter=%d)',lower(video),max_iter);
                if strcmp(spike_type, 'NAOMi')
                    if nbin == 1 % NAOMi
                        folder = sprintf('traces_%s_%s%s%s%s%s',method,video,part1,part2,part3,[addon,addon_nobin]);
                        scores = sprintf('%s\\scores_split_%s_%s_%sVideo_%s_Sigma%s%s%s%s%s.mat',dir_scores,...
                            method,simu_opt,video,sigma_from,part1,part2,part3,[addon,addon_nobin],baseline_std);
                    else
                        folder = sprintf('traces_%s_%s_%s%d%s%s%s%s',method,video,bin_option,nbin,part1,part2,part3,addon);
                        scores = sprintf('%s\\scores_split_%s_%s_%sVideo_%s_Sigma_%s%d%s%s%s%s%s.mat',dir_scores,...
                            method,simu_opt,video,sigma_from,bin_option,nbin,part1,part2,part3,addon,baseline_std);
                    end
                else
                    if nbin == 1 % ABO and 1p
                        folder = sprintf('traces_%s_%s%s%s%s%s',method,video,part1,part2,part3,[addon,addon_nobin]);
                        scores = sprintf('%s\\scores_split_%s_%sVideo%s%s%s%s_%sSigma%s.mat',dir_scores,...
                            method,video,part1,part2,part3,[addon,addon_nobin],sigma_from,baseline_std);
                    else
                        folder = sprintf('traces_%s_%s_%s%d%s%s%s%s',method,video,bin_option,nbin,part1,part2,part3,addon);
                        scores = sprintf('%s\\scores_split_%s_%sVideo_%s%d%s%s%s%s_%sSigma%s.mat',dir_scores,...
                            method,video,bin_option,nbin,part1,part2,part3,addon,sigma_from,baseline_std);
                    end
                end
        %         folder = sprintf('traces_ours');
                dir_FISSA = fullfile(dir_traces,folder);
                load([dir_FISSA,'\Table_time.mat'],'list_alpha','Table_time')
                list_alpha_all_time{bid,oid,vid} = list_alpha;
                Table_time_all{bid,oid,vid} = Table_time;

                load(scores,'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
                list_recall_all{bid,oid,vid} = list_recall;
                list_precision_all{bid,oid,vid} = list_precision;
                list_F1_all{bid,oid,vid} = list_F1;
                list_thred_ratio_all{bid,oid,vid} = list_thred_ratio;
                list_alpha_all{bid,oid,vid} = list_alpha;
            end
        end
    end
    if strcmp(spike_type, 'NAOMi')
        save(sprintf('%s\\timing_split_BinUnmix%s_100.mat',dir_scores,addon),...
            'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option',...
            'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all'); % 
    else
        save(sprintf('%s\\timing_split_BinUnmix%s_100.mat',dir_scores,addon),...
            'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option',...
            'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all'); % 
    end
end