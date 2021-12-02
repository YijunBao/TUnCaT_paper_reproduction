clear;
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
color_many = distinguishable_colors(16);
dir_scores='..\results\ABO\evaluation\';
% dir_scores='..\evaluation\ABO\';

%% Plot F1 vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+2;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;
size_mean = 20;
size_each = 30;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([7,6],:);
% colors = color_many([5,8],:);
colors = [color(5,:);color_many(5,:)];
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load([dir_scores,'\F1_split_fix_float_alpha_ksd-psd.mat'])
num_addon = length(list_addon);
clear h he hs

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    for addid = 1:num_addon
        addon = list_addon{addid};
        list_alpha = Table_list_alpha{addid,vid};
        recall_CV = Table_recall_CV{addid,vid};
        precision_CV = Table_precision_CV{addid,vid};
        F1_CV = Table_F1_CV{addid,vid};
        thred_ratio_CV = Table_thred_ratio_CV{addid,vid};
        list_alpha_select = find((list_alpha<=max_alpha) & (list_alpha>=min_alpha));
        list_alpha_select = list_alpha_select(1:step:end);
        list_alpha = list_alpha(list_alpha_select);
        list_F1 = F1_CV(:,list_alpha_select,:);
        num_alpha = length(list_alpha);
        mean_F1 = squeeze(mean(list_F1,1));
        std_F1 = squeeze(std(list_F1,1,1));
%         [max_F1, pos] = max(mean_F1,[],2);
%         num_alpha = length(list_alpha);

        h(num_addon+1-addid) = plot(list_alpha',mean_F1,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',colors(addid,:)); % 
        he(num_addon+1-addid) = errorbar(list_alpha',mean_F1,std_F1,std_F1,'LineWidth',1,'CapSize',10,'HandleVisibility','off','Color',colors(addid,:)); % 
        hs(num_addon+1-addid) = scatter(reshape(repmat(list_alpha',size(list_F1,1),1),1,[]),list_F1(:),size_each,colors(addid,:),'x');  % ,'HandleVisibility','off'  
        yl2 = [0.6,1.0];
%         yl2(1) = floor(min(min(mean_F1-std_F1))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_F1+std_F1))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = 0.98;
%         list_y = max(max(mean_F1+std_F1))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
    set(gca,'Children',[h, hs, he])
    for aid = 1:num_alpha % list_alpha_select'
        alpha_id = list_alpha_select(aid);
        p_sign = signrank(Table_F1_CV{1,vid}(:,alpha_id,:),Table_F1_CV{2,vid}(:,alpha_id,:)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y+step_ns;
        else
            y = list_y;
        end
        text(list_alpha(aid),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center');
    end
    ylabel('{\itF}_1');
    xlabel('\alpha');
    set(gca,'FontSize',17);
    set(gca,'XScale','log');
    xlim([10.^floor(log10(min_alpha)-1),10^ceil(log10(max_alpha))]);
    xticks(10.^(floor(log10(min_alpha)-1):ceil(log10(max_alpha))));
    legend({'Floating \alpha individual','Fixed \alpha individual',...
        'Floating \alpha mean','Fixed \alpha mean'},'Location','SouthWest');
    title([video,' videos'])
%     saveas(gcf,sprintf('ABO float fix alpha split, %s Video, alpha=%s-%s.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO float fix alpha F1, %s Video.png',video));
end

%% Plot time vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+4;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;
size_mean = 20;
size_each = 30;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([7,6],:);
% colors = color_many([5,8],:);
colors = [color(5,:);color_many(5,:)];
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load([dir_scores,'\Time_alpha_ABO.mat'],'list_alpha_all','Table_time_all','list_video','list_addon')
num_addon = length(list_addon);
clear h he hs

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    Table_time_vid = cell(length(list_addon),1);
    for addid = 1:num_addon
        list_alpha = list_alpha_all{addid,vid};
        Table_time = Table_time_all{addid,vid};
        Table_time = Table_time + Table_time(:,end);
        Table_time = Table_time(:,1:end-1);
        Table_time_vid{addid} = Table_time;
        list_alpha_select = find((list_alpha<=max_alpha) & (list_alpha>=min_alpha));
        list_alpha_select = list_alpha_select(1:step:end);
        list_alpha = list_alpha(list_alpha_select);
        Table_time = Table_time(:,list_alpha_select);
        num_alpha = length(list_alpha);
        mean_time = squeeze(mean(Table_time,1));
        std_time = squeeze(std(Table_time,1,1));
%         [max_F1, pos] = max(mean_F1,[],2);
%         num_alpha = length(list_alpha);

        h(num_addon+1-addid) = plot(list_alpha',mean_time,'.-','MarkerSize',size_mean,'LineWidth',2,'Color',colors(addid,:)); % 
        he(num_addon+1-addid) = errorbar(list_alpha',mean_time,std_time,std_time,'LineWidth',1,'CapSize',10,'HandleVisibility','off','Color',colors(addid,:)); % 
        hs(num_addon+1-addid) = scatter(reshape(repmat(list_alpha,size(Table_time,1),1),1,[]),Table_time(:),size_each,colors(addid,:),'x');  % ,'HandleVisibility','off'  
        yl2 = [0,120];
%         yl2(1) = floor(min(min(mean_time-std_time))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_time+std_time))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(Table_time(:)))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
    set(gca,'Children',[h, hs, he])
    for aid = 1:num_alpha % list_alpha_select'
        alpha_id = list_alpha_select(aid);
        p_sign = signrank(Table_time_vid{1}(:,alpha_id),Table_time_vid{2}(:,alpha_id)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y+step_ns;
        else
            y = list_y;
        end
        text(list_alpha(aid),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center');
    end
    ylabel('Processing time (s)');
    xlabel('\alpha');
    set(gca,'FontSize',17);
    set(gca,'XScale','log');
    xlim([10.^floor(log10(min_alpha)-1),10^ceil(log10(max_alpha))]);
    xticks(10.^(floor(log10(min_alpha)-1):ceil(log10(max_alpha))));
    le = legend({'Floating \alpha individual','Fixed \alpha individual',...
        'Floating \alpha mean','Fixed \alpha mean'}); 
    if list_y < 90
        set(le,'Position',[0.5,0.7,0.4,0.2]);
    else
        set(le,'Position',[0.5,0.55,0.4,0.2]);
    end
    title([video,' videos'])
%     saveas(gcf,sprintf('ABO float fix alpha time, %s Video, alpha=%s-%s.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO float fix alpha time, %s Video.png',video));
end
