% clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
list_spike_type = {'ABO','NAOMi','1p'};
for ind = 1:length(list_spike_type)
    spike_type = list_spike_type{ind};
    dir_video = ['..\data\',spike_type];
%     if strcmp(spike_type,'ABO')
%         % dir_video='D:\ABO\20 percent 200';
%         list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%                      '510214538';'524691284';'527048992';'531006860';'539670003'};
%     elseif strcmp(spike_type,'1p')
%         % dir_video='E:\OnePhoton videos\cropped videos\';
%         list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%             'c25_163_267','c27_114_176','c28_161_149',...
%             'c25_123_348','c27_122_121','c28_163_244'};
%     elseif strcmp(spike_type,'NAOMi')
%         % simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; % _NA0.4,0.3
%         % dir_video=['F:\NAOMi\',simu_opt,'\']; % _hasStart
%         list_Exp_ID=arrayfun(@(x) ['Video_',num2str(x)], 0:9, 'UniformOutput', false);
%     end
    
    %% common
    addon = '_novideounmix_r2'; % '_eps=0.1'; % 
    addon_nobin = '_mixout';
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
                        scores = sprintf('.\\%s\\scores_split_%s_%s_%sVideo_%s_compSigma%s%s%s%s%s.mat',spike_type,...
                            method,simu_opt,video,sigma_from,part1,part2,part3,[addon,addon_nobin],baseline_std);
                    else
                        folder = sprintf('traces_%s_%s_%s%d%s%s%s%s',method,video,bin_option,nbin,part1,part2,part3,addon);
                        scores = sprintf('.\\%s\\scores_split_%s_%s_%sVideo_%s_compSigma_%s%d%s%s%s%s%s.mat',spike_type,...
                            method,simu_opt,video,sigma_from,bin_option,nbin,part1,part2,part3,addon,baseline_std);
                    end
                else
                    if nbin == 1 % ABO and 1p
                        folder = sprintf('traces_%s_%s%s%s%s%s',method,video,part1,part2,part3,[addon,addon_nobin]);
                        scores = sprintf('.\\%s\\scores_split_%s_%sVideo%s%s%s%s_%sSigma%s.mat',spike_type,...
                            method,video,part1,part2,part3,[addon,addon_nobin],sigma_from,baseline_std);
                    else
                        folder = sprintf('traces_%s_%s_%s%d%s%s%s%s',method,video,bin_option,nbin,part1,part2,part3,addon);
                        scores = sprintf('.\\%s\\scores_split_%s_%sVideo_%s%d%s%s%s%s_%sSigma%s.mat',spike_type,...
                            method,video,bin_option,nbin,part1,part2,part3,addon,sigma_from,baseline_std);
                    end
                end
        %         folder = sprintf('traces_ours');
                dir_FISSA = fullfile(dir_video,folder);
                load([dir_FISSA,'\Table_time.mat'],'list_alpha','Table_time')
                list_alpha_all_time{bid,oid,vid} = list_alpha;
                Table_time_all{bid,oid,vid} = Table_time;

                load(scores,'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
                list_recall_all{bid,oid,vid} = list_recall;
                list_precision_all{bid,oid,vid} = list_precision;
                list_F1_all{bid,oid,vid} = list_F1;
                list_thred_ratio_all{bid,oid,vid} = list_thred_ratio;
                list_alpha_all{bid,oid,vid} = list_alpha;

                if strcmp(spike_type, 'NAOMi')
                    load(scores,'list_corr_unmix');
                    list_corr_unmix_all{bid,oid,vid} = list_corr_unmix;
                end
            end
        end
    end
    if strcmp(spike_type, 'NAOMi')
        save(sprintf('%s\\timing_split_BinUnmix%s_100.mat',spike_type,addon),...
            'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option','list_corr_unmix_all',...
            'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all'); % 
    else
        save(sprintf('%s\\timing_split_BinUnmix%s_100.mat',spike_type,addon),...
            'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option',...
            'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all'); % 
    end
end