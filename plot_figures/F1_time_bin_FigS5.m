clear;
addpath(genpath('..\evaluation'));
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];

%% ABO or 1p
spike_type = 'ABO'; % {'exclude','only','include'};
addon = ''; % '_eps=0.1'; % 
dir_scores=['..\results\',spike_type,'\evaluation\'];
% dir_scores=['..\evaluation\',spike_type,'\'];

list_y = [0.855,0.975];
list_y_time = [60,15];
step_ns = 0.0015;
step_ns_time = 1;
ylim_set = [0.85,1];

size_mean = 20;
size_each = 30;

% sigma_from = 'Unmix'; % 'Raw_comp'; % {'Unmix'}; % 
% list_video= {'Raw'}; % 'Raw','SNR'
% baseline_std = '_ksd-psd'; % '', 
% method = 'ours'; % {'FISSA','ours'}
% list_bin_option = {'sum','mean','downsample'}; % 
load([dir_scores,'\timing_split_BinUnmix',addon,'_100.mat'],...
    'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option',...
    'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all')
list_bin_option = list_bin_option(end);
list_nbin = list_nbin';
list_video_legend = cellfun(@(x) [x,' videos'],list_video, 'UniformOutput',false);
list_alpha_all_time = list_alpha_all_time(:,end,:);
Table_time_all = Table_time_all(:,end,:);
list_recall_all = list_recall_all(:,end,:);
list_precision_all = list_precision_all(:,end,:);
list_F1_all = list_F1_all(:,end,:);
list_thred_ratio_all = list_thred_ratio_all(:,end,:);
list_alpha_all = list_alpha_all(:,end,:);

num_video = length(list_video);
num_bin_option = length(list_bin_option);
num_nbin = length(list_nbin);
num_Exp = size(list_F1_all{1},1);
xtext = 'Downsampling ratio';
list_bin_option = cellfun(@(x) [upper(x(1)),x(2:end)], list_bin_option, 'UniformOutput',false);

% step_score = 0.02;
% step_time = 2; % 1.4; % 
% step_ns_score = 0.00;
% step_ns_time = 1;
% grid_score = 0.05;
% grid_time = 5;
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14];
list_p_cutoff = [0.05, 0.005, 0];

[recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV] = deal(zeros(num_Exp,num_nbin,num_bin_option,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_nbin,num_bin_option,num_video));

for vid = 1:num_video
    video = list_video(vid);
for bid = 1:num_nbin
%     nbin = list_nbin(bid);
    for oid = 1:num_bin_option
%         bin_option = num_bin_option(oid);
    list_recall = list_recall_all{bid,oid,vid};
    list_precision = list_precision_all{bid,oid,vid};
    list_F1 = list_F1_all{bid,oid,vid};
    Table_time = Table_time_all{bid,oid,vid};
    list_thred_ratio = list_thred_ratio_all{bid,oid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{bid,oid,vid} = zeros(n1,n3);
        Table_precision_CV{bid,oid,vid} = zeros(n1,n3);
        Table_F1_CV{bid,oid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{bid,oid,vid};
        list_alpha_time = list_alpha_all_time{bid,oid,vid};
    else
        Table_recall_CV{bid,oid,vid} = squeeze(list_recall);
        Table_precision_CV{bid,oid,vid} = squeeze(list_precision);
        Table_F1_CV{bid,oid,vid} = squeeze(list_F1);
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
        recall_CV(CV,bid,oid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,bid,oid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,bid,oid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,bid,oid,vid) = alpha;
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_thred_ratio);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,bid,oid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
            Table_recall_CV{bid,oid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{bid,oid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{bid,oid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
        else
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,bid,oid,vid) = Table_time(CV);
        end
    end
    end
end
end

%% F1 and time 
figure('Position',[900,100,450,450]);
clear ax;
% ax(1) = subplot(1,2,1);
hold on;
for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    errorbar(list_nbin,F1,F1_err,F1_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),F1_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    video = list_video{vid};

    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    h=plot(list_nbin,F1,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
%     yl=get(gca,'Ylim');
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(F1-F1_err))/grid_score)*grid_score;
%     yl2(2) = floor(max(max(F1+F1_err))/grid_score)*grid_score;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(F1+F1_err))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_score)*grid_score;
%     ylim(yl2);
    for jj = 2:num_nbin
        p_sign = signrank(F1_all(:,jj),F1_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y(vid)+step_ns;
        else
            y = list_y(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end
    ylabel('{\itF}_1');
    xlabel(xtext)
    % xlim()
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
end
ylim(ylim_set);
saveas(gcf,[spike_type,' F1 bin downsample p.png']);
% saveas(gcf,[spike_type,' F1 bin downsample p.emf']);

figure('Position',[900,100+450,450,450]);
hold on;
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid));
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    errorbar(list_nbin,time,time_err,time_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid)); % alpha_CV
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),time_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid)); % alpha_CV
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    h=plot(list_nbin,time,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
%     set(gca,'YScale','log');
    ylabel('Processing time (s)');
    xlabel(xtext)
    % xlim()
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
%     yl=get(gca,'Ylim');
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(time-time_err))/grid_time)*grid_time;
%     yl2(2) = floor(max(max(time+time_err))/grid_time)*grid_time;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(time(2:end)+time_err(2:end)))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_time)*grid_time;
%     ylim(yl2);
%     time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time);
%     errorbar(list_nbin,time,time_err_low,time_err,...
%         'LineWidth',1,'Color',h.Color,'HandleVisibility','off');
    for jj = 2:num_nbin
        p_sign = signrank(time_all(:,jj),time_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y_time(vid)+step_ns_time;
        else
            y = list_y_time(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end

end
%     title_str = [video,' NAOMi videos with different frequency'];
%     suptitle(title_str);
% ylim([15,50]);
saveas(gcf,[spike_type,' time bin downsample p.png']);
% saveas(gcf,[spike_type,' time bin downsample p.emf']);

%% alpha and thresh_ratio
% figure('Position',[100,100+450*(vid-1),900,450]);
% clear ax;
% for vid = 1:num_video
%     video = list_video{vid};
%     % disp(data(:,3)');
% 
%     F1_all = squeeze(thred_ratio_CV(:,:,:,vid));
%     F1 = squeeze(mean(F1_all,1))';
%     F1_err = squeeze(std(F1_all,1,1))';
%     ax(1) = subplot(1,2,2);
%     hold on;
%     h=plot(list_nbin,F1,'LineWidth',2,'Color',color(5+vid,:));
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,F1,F1_err,F1_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     ylabel('SNR threshold');
%     xlabel(xtext)
%     % xlim()
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
% 
%     time_all = squeeze(alpha_CV(:,:,:,vid)); % alpha_CV
%     time = squeeze(mean(time_all,1))';
%     time_err = squeeze(std(time_all,1,1))';
%     ax(2) = subplot(1,2,1);
%     hold on;
%     h=plot(list_nbin,time,'LineWidth',2,'Color',color(5+vid,:));
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     set(gca,'YScale','log');
%     ylabel('\alpha');
%     xlabel(xtext)
%     % xlim()
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,time,time_err,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     yl=get(gca,'Ylim');
%     for oid = 1:num_bin_option
%         time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time(oid,:));
%         errorbar(list_nbin,time,time_err_low,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
% 
% %     title_str = [video,' NAOMi videos with different frequency'];
% %     suptitle(title_str);
%     saveas(gcf,[spike_type,' alpha bin downsample.png']);
% end



%% 1p
spike_type = '1p'; % {'exclude','only','include'};
dir_scores=['..\results\',spike_type,'\evaluation\'];
% dir_scores=['..\evaluation\',spike_type,'\'];
addon = ''; % '_eps=0.1'; % 

list_y = [0.62,0.93];
list_y_time = [25,2];
step_ns = 0.005;
step_ns_time = 1;
ylim_set = [0.60,0.95];

size_mean = 20;
size_each = 30;

% sigma_from = 'Unmix'; % 'Raw_comp'; % {'Unmix'}; % 
% list_video= {'Raw'}; % 'Raw','SNR'
% baseline_std = '_ksd-psd'; % '', 
% method = 'ours'; % {'FISSA','ours'}
% list_bin_option = {'sum','mean','downsample'}; % 
load([dir_scores,'\timing_split_BinUnmix',addon,'_100.mat'],...
    'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option',...
    'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all')
list_bin_option = list_bin_option(end);
list_nbin = list_nbin';
list_video_legend = cellfun(@(x) [x,' videos'],list_video, 'UniformOutput',false);
list_alpha_all_time = list_alpha_all_time(:,end,:);
Table_time_all = Table_time_all(:,end,:);
list_recall_all = list_recall_all(:,end,:);
list_precision_all = list_precision_all(:,end,:);
list_F1_all = list_F1_all(:,end,:);
list_thred_ratio_all = list_thred_ratio_all(:,end,:);
list_alpha_all = list_alpha_all(:,end,:);

num_video = length(list_video);
num_bin_option = length(list_bin_option);
num_nbin = length(list_nbin);
num_Exp = size(list_F1_all{1},1);
xtext = 'Downsampling ratio';
list_bin_option = cellfun(@(x) [upper(x(1)),x(2:end)], list_bin_option, 'UniformOutput',false);

% step_score = 0.02;
% step_time = 2; % 1.4; % 
% step_ns_score = 0.00;
% step_ns_time = 1;
% grid_score = 0.05;
% grid_time = 5;
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14];
list_p_cutoff = [0.05, 0.005, 0];

[recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV] = deal(zeros(num_Exp,num_nbin,num_bin_option,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_nbin,num_bin_option,num_video));

for vid = 1:num_video
    video = list_video(vid);
for bid = 1:num_nbin
%     nbin = list_nbin(bid);
    for oid = 1:num_bin_option
%         bin_option = num_bin_option(oid);
    list_recall = list_recall_all{bid,oid,vid};
    list_precision = list_precision_all{bid,oid,vid};
    list_F1 = list_F1_all{bid,oid,vid};
    Table_time = Table_time_all{bid,oid,vid};
    list_thred_ratio = list_thred_ratio_all{bid,oid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{bid,oid,vid} = zeros(n1,n3);
        Table_precision_CV{bid,oid,vid} = zeros(n1,n3);
        Table_F1_CV{bid,oid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{bid,oid,vid};
        list_alpha_time = list_alpha_all_time{bid,oid,vid};
    else
        Table_recall_CV{bid,oid,vid} = squeeze(list_recall);
        Table_precision_CV{bid,oid,vid} = squeeze(list_precision);
        Table_F1_CV{bid,oid,vid} = squeeze(list_F1);
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
        recall_CV(CV,bid,oid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,bid,oid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,bid,oid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,bid,oid,vid) = alpha;
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_thred_ratio);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,bid,oid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
            Table_recall_CV{bid,oid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{bid,oid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{bid,oid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
        else
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,bid,oid,vid) = Table_time(CV);
        end
    end
    end
end
end

%% F1 and time 
figure('Position',[900,100,450,450]);
clear ax;
% ax(1) = subplot(1,2,1);
hold on;
for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    errorbar(list_nbin,F1,F1_err,F1_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),F1_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    video = list_video{vid};

    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    h=plot(list_nbin,F1,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
%     yl=get(gca,'Ylim');
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(F1-F1_err))/grid_score)*grid_score;
%     yl2(2) = floor(max(max(F1+F1_err))/grid_score)*grid_score;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(F1+F1_err))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_score)*grid_score;
%     ylim(yl2);
    for jj = 2:num_nbin
        p_sign = signrank(F1_all(:,jj),F1_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y(vid)+step_ns;
        else
            y = list_y(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end
    ylabel('{\itF}_1');
    xlabel(xtext)
    % xlim()
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
end
ylim(ylim_set);
saveas(gcf,[spike_type,' F1 bin downsample p.png']);
% saveas(gcf,[spike_type,' F1 bin downsample p.emf']);

figure('Position',[900,100+450,450,450]);
hold on;
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid));
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    errorbar(list_nbin,time,time_err,time_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid)); % alpha_CV
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),time_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid)); % alpha_CV
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    h=plot(list_nbin,time,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
%     set(gca,'YScale','log');
    ylabel('Processing time (s)');
    xlabel(xtext)
    % xlim()
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
%     yl=get(gca,'Ylim');
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(time-time_err))/grid_time)*grid_time;
%     yl2(2) = floor(max(max(time+time_err))/grid_time)*grid_time;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(time(2:end)+time_err(2:end)))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_time)*grid_time;
%     ylim(yl2);
%     time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time);
%     errorbar(list_nbin,time,time_err_low,time_err,...
%         'LineWidth',1,'Color',h.Color,'HandleVisibility','off');
    for jj = 2:num_nbin
        p_sign = signrank(time_all(:,jj),time_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y_time(vid)+step_ns_time;
        else
            y = list_y_time(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end

end
%     title_str = [video,' NAOMi videos with different frequency'];
%     suptitle(title_str);
% ylim([15,50]);
saveas(gcf,[spike_type,' time bin downsample p.png']);
% saveas(gcf,[spike_type,' time bin downsample p.emf']);

%% alpha and thresh_ratio
% figure('Position',[100,100+450*(vid-1),900,450]);
% clear ax;
% for vid = 1:num_video
%     video = list_video{vid};
%     % disp(data(:,3)');
% 
%     F1_all = squeeze(thred_ratio_CV(:,:,:,vid));
%     F1 = squeeze(mean(F1_all,1))';
%     F1_err = squeeze(std(F1_all,1,1))';
%     ax(1) = subplot(1,2,2);
%     hold on;
%     h=plot(list_nbin,F1,'LineWidth',2,'Color',color(5+vid,:));
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,F1,F1_err,F1_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     ylabel('SNR threshold');
%     xlabel(xtext)
%     % xlim()
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
% 
%     time_all = squeeze(alpha_CV(:,:,:,vid)); % alpha_CV
%     time = squeeze(mean(time_all,1))';
%     time_err = squeeze(std(time_all,1,1))';
%     ax(2) = subplot(1,2,1);
%     hold on;
%     h=plot(list_nbin,time,'LineWidth',2,'Color',color(5+vid,:));
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     set(gca,'YScale','log');
%     ylabel('\alpha');
%     xlabel(xtext)
%     % xlim()
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,time,time_err,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     yl=get(gca,'Ylim');
%     for oid = 1:num_bin_option
%         time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time(oid,:));
%         errorbar(list_nbin,time,time_err_low,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
% 
% %     title_str = [video,' NAOMi videos with different frequency'];
% %     suptitle(title_str);
%     saveas(gcf,[spike_type,' alpha bin downsample.png']);
% end



%% NAOMi
spike_type = 'NAOMi'; % {'exclude','only','include'};
dir_scores=['..\results\',spike_type,'\evaluation\'];
% dir_scores=['..\evaluation\',spike_type,'\'];
addon = ''; % '_eps=0.1'; % 
list_y = [0.52,0.98];
list_y_corr = [0.38,0.98];
list_y_time = [3,9.5];
step_ns = 0.01;
step_ns_time = 0.2;

size_mean = 20;
size_each = 30;

% sigma_from = 'Unmix'; % 'Raw_comp'; % {'Unmix'}; % 
% list_video= {'Raw'}; % 'Raw','SNR'
% baseline_std = '_ksd-psd'; % '', 
% method = 'ours'; % {'FISSA','ours'}
% list_bin_option = {'sum','mean','downsample'}; % 
load([dir_scores,'\timing_split_BinUnmix',addon,'_100.mat'],...
    'list_alpha_all_time','Table_time_all','list_nbin','list_video','list_bin_option','list_corr_unmix_all',...
    'list_recall_all','list_precision_all','list_F1_all','list_thred_ratio_all','list_alpha_all')
list_bin_option = list_bin_option(end);
list_nbin = list_nbin';
list_video_legend = cellfun(@(x) [x,' videos'],list_video, 'UniformOutput',false);
list_alpha_all_time = list_alpha_all_time(:,end,:);
Table_time_all = Table_time_all(:,end,:);
list_recall_all = list_recall_all(:,end,:);
list_precision_all = list_precision_all(:,end,:);
list_F1_all = list_F1_all(:,end,:);
list_thred_ratio_all = list_thred_ratio_all(:,end,:);
list_alpha_all = list_alpha_all(:,end,:);

num_video = length(list_video);
num_bin_option = length(list_bin_option);
num_nbin = length(list_nbin);
num_Exp = size(list_F1_all{1},1);
xtext = 'Downsampling ratio';
list_bin_option = cellfun(@(x) [upper(x(1)),x(2:end)], list_bin_option, 'UniformOutput',false);

% step_score = 0.02;
% step_time = 2; % 1.4; % 
% step_ns_score = 0.00;
% step_ns_time = 1;
% grid_score = 0.05;
% grid_time = 2;
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14];
list_p_cutoff = [0.05, 0.005, 0];

[recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV, corr_CV] ...
    = deal(zeros(num_Exp,num_nbin,num_bin_option,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_nbin,num_bin_option,num_video));

for vid = 1:num_video
    video = list_video(vid);
for bid = 1:num_nbin
%     nbin = list_nbin(bid);
    for oid = 1:num_bin_option
%         bin_option = num_bin_option(oid);
    list_recall = list_recall_all{bid,oid,vid};
    list_precision = list_precision_all{bid,oid,vid};
    list_F1 = list_F1_all{bid,oid,vid};
    Table_time = Table_time_all{bid,oid,vid};
    list_thred_ratio = list_thred_ratio_all{bid,oid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{bid,oid,vid} = zeros(n1,n3);
        Table_precision_CV{bid,oid,vid} = zeros(n1,n3);
        Table_F1_CV{bid,oid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{bid,oid,vid};
        list_alpha_time = list_alpha_all_time{bid,oid,vid};
    else
        Table_recall_CV{bid,oid,vid} = squeeze(list_recall);
        Table_precision_CV{bid,oid,vid} = squeeze(list_precision);
        Table_F1_CV{bid,oid,vid} = squeeze(list_F1);
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
        recall_CV(CV,bid,oid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,bid,oid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,bid,oid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,bid,oid,vid) = alpha;
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_thred_ratio);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,bid,oid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
            Table_recall_CV{bid,oid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{bid,oid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{bid,oid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
        else
            thred_ratio_CV(CV,bid,oid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,bid,oid,vid) = Table_time(CV);
        end
    end
    end
end
end

%% F1 and time
figure('Position',[500,100,450,450]);
clear ax;
% ax(1) = subplot(1,3,1);
hold on;

for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    errorbar(list_nbin,F1,F1_err,F1_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    F1_all = squeeze(F1_CV(:,:,:,vid));
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),F1_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    video = list_video{vid};
    % disp(data(:,3)');

    F1_all = squeeze(F1_CV(:,:,:,vid));
    F1 = squeeze(mean(F1_all,1))';
    F1_err = squeeze(std(F1_all,1,1))';
    h=plot(list_nbin,F1,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(F1-F1_err))/grid_score)*grid_score;
%     yl2(2) = floor(max(max(F1+F1_err))/grid_score)*grid_score;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(F1+F1_err))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_score)*grid_score;
%     ylim(yl2);
    for jj = 2:num_nbin
        p_sign = signrank(F1_all(:,jj),F1_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y(vid)+step_ns;
        else
            y = list_y(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end
    ylabel('{\itF}_1');
    xlabel(xtext)
    % xlim()
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
end
% ylim([0.65,1]);
saveas(gcf,[spike_type,' F1 bin downsample p.png']);
% saveas(gcf,[spike_type,' F1 bin downsample p.emf']);

figure('Position',[500,550,450,450]);
hold on;
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid));
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    errorbar(list_nbin,time,time_err,time_err,'CapSize',10,...
        'LineWidth',1,'Color',color(5+vid,:),'HandleVisibility','off');
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid));
    scatter(reshape(repmat(list_nbin',num_Exp,1),1,[]),time_all(:),size_each,color(5+vid,:),'x','HandleVisibility','off');    
end
for vid = 1:num_video
    time_all = squeeze(time_CV(:,:,:,vid)); % alpha_CV
    time = squeeze(mean(time_all,1))';
    time_err = squeeze(std(time_all,1,1))';
    h=plot(list_nbin,time,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',color(5+vid,:));
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
%     set(gca,'YScale','log');
    ylabel('Processing time (s)');
    xlabel(xtext)
    % xlim()
    legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
    yl=get(gca,'Ylim');
%     yl2 = [0,0];
%     yl2(1) = floor(min(min(time-time_err))/grid_time)*grid_time;
%     yl2(2) = floor(max(max(time+time_err))/grid_time)*grid_time;
%     step = (yl2(2)-yl2(1))/10;
%     step_ns = step/3;
%     list_y = max(max(time(:,2:end)+time_err(:,2:end)))+step*(1:num_bin_option);
%     yl2(2) = ceil(list_y(end)/grid_time)*grid_time;
%     ylim(yl2);
%     time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time);
%     errorbar(list_nbin,time,time_err_low,time_err,...
%         'LineWidth',1,'Color',h.Color,'HandleVisibility','off');
    ylim([0,20]);
%     time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time);
%     errorbar(list_nbin,time,time_err_low,time_err,...
%         'LineWidth',1,'Color',h.Color,'HandleVisibility','off');
    for jj = 2:num_nbin
        p_sign = signrank(time_all(:,jj),time_all(:,1)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y_time(vid)+step_ns_time;
        else
            y = list_y_time(vid);
        end
        text(list_nbin(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center','Color',h.Color);
    end
end
%     title_str = [video,' NAOMi videos with different frequency'];
%     suptitle(title_str);
saveas(gcf,[spike_type,' time bin downsample videos p.png']);
% saveas(gcf,[spike_type,' time bin downsample videos p.emf']);

%% alpha and thresh_ratio
% figure('Position',[10,100+450*(vid-1),900,400]);
% clear ax;
% for vid = 1:num_video
%     video = list_video{vid};
%     % disp(data(:,3)');
% 
%     F1_all = squeeze(thred_ratio_CV(:,:,:,vid));
%     F1 = squeeze(mean(F1_all,1))';
%     F1_err = squeeze(std(F1_all,1,1))';
%     ax(1) = subplot(1,2,2);
%     hold on;
%     h=plot(list_nbin,F1,'LineWidth',2,'Color',color(5+vid,:));
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,F1,F1_err,F1_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     ylabel('SNR threshold');
%     xlabel(xtext)
%     % xlim()
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
% 
%     time_all = squeeze(alpha_CV(:,:,:,vid)); % alpha_CV
%     time = squeeze(mean(time_all,1))';
%     time_err = squeeze(std(time_all,1,1))';
%     ax(2) = subplot(1,2,1);
%     hold on;
%     h=plot(list_nbin,time,'LineWidth',2,'Color',color(5+vid,:));
%     set(gca,'FontSize',14);
%     set(gca,'XScale','log');
%     set(gca,'YScale','log');
%     ylabel('\alpha');
%     xlabel(xtext)
%     % xlim()
%     legend(list_video_legend,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
%     for oid = 1:num_bin_option
%         errorbar(list_nbin,time,time_err,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
%     yl=get(gca,'Ylim');
%     for oid = 1:num_bin_option
%         time_err_low = time_err.*(time_err<=time) + (time-yl(1)).*(time_err>time);
%         errorbar(list_nbin,time,time_err_low,time_err,...
%             'LineWidth',1,'Color',h(oid).Color,'HandleVisibility','off');
%     end
% 
% %     title_str = [video,' NAOMi videos with different frequency'];
% %     suptitle(title_str);
%     saveas(gcf,[spike_type,' alpha bin downsample.png']);
% %     saveas(gcf,[spike_type,' alpha bin downsample.emf']);
% end
