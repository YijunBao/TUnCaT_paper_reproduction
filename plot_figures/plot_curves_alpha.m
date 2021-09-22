color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
color_many = distinguishable_colors(16);

%% Plot F1 vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+2;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([6,7],:);
colors = color_many([5,8],:);
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load('include\F1_split_fix_float_alpha_ksd-psd.mat')

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    for addid = 1:length(list_addon)
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

        h=plot(list_alpha',mean_F1,'.-','MarkerSize',18,'LineWidth',2,'Color',colors(addid,:)); % 
        errorbar(list_alpha',mean_F1,std_F1,std_F1,'LineWidth',1,'HandleVisibility','off','Color',colors(addid,:)); % 
        yl2 = [0.7,1.0];
%         yl2(1) = floor(min(min(mean_F1-std_F1))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_F1+std_F1))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(mean_F1+std_F1))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
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
    legend({'Floating \alpha','Fixed \alpha'},'Location','SouthWest');
    title([video,' videos'])
    saveas(gcf,sprintf('ABO float fix alpha split, %s Video, alpha=%s-%s 0910.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO float fix alpha split, %s Video, alpha=%s-%s 0910.png',video,num2str(min_alpha),num2str(max_alpha)));
end

%% Plot time vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+4;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([6,7],:);
colors = color_many([5,8],:);
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load('include\Time_alpha_ABO.mat','list_alpha_all','Table_time_all','list_video','list_addon')

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    Table_time_vid = cell(length(list_addon),1);
    for addid = 1:length(list_addon)
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

        h=plot(list_alpha',mean_time,'.-','MarkerSize',18,'LineWidth',2,'Color',colors(addid,:)); % 
        errorbar(list_alpha',mean_time,std_time,std_time,'LineWidth',1,'HandleVisibility','off','Color',colors(addid,:)); % 
        yl2 = [0,100];
%         yl2(1) = floor(min(min(mean_time-std_time))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_time+std_time))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(mean_time+std_time))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
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
    legend({'Floating \alpha','Fixed \alpha'},'Location','NorthEast');
    title([video,' videos'])
    saveas(gcf,sprintf('ABO float fix alpha time, %s Video, alpha=%s-%s 0910.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO float fix alpha time, %s Video, alpha=%s-%s 0910.png',video,num2str(min_alpha),num2str(max_alpha)));
end

%% Plot n_iter vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+4;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([6,7],:);
colors = color_many([5,8],:);
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load('include\Niter_alpha_ABO.mat','list_alpha_all','list_n_iter_all','list_video','list_addon')

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    list_n_iter_vid = cell(length(list_addon),1);
    for addid = 1:length(list_addon)
        list_alpha = list_alpha_all{addid,vid};
        list_n_iter = list_n_iter_all{addid,vid};
        list_n_iter_vid{addid} = list_n_iter;
        list_alpha_select = find((list_alpha<=max_alpha) & (list_alpha>=min_alpha));
        list_alpha_select = list_alpha_select(1:step:end);
        list_alpha = list_alpha(list_alpha_select);
        list_n_iter = list_n_iter(:,list_alpha_select);
        num_alpha = length(list_alpha);
        mean_n_iter = squeeze(mean(list_n_iter,1));
        std_n_iter = squeeze(std(list_n_iter,1,1));
%         [max_F1, pos] = max(mean_F1,[],2);
%         num_alpha = length(list_alpha);

        h=plot(list_alpha',mean_n_iter,'.-','MarkerSize',18,'LineWidth',2,'Color',colors(addid,:)); % 
        errorbar(list_alpha',mean_n_iter,std_n_iter,std_n_iter,'LineWidth',1,'HandleVisibility','off','Color',colors(addid,:)); % 
        yl2 = [0,12000];
%         yl2(1) = floor(min(min(mean_n_iter-std_n_iter))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_n_iter+std_n_iter))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(mean_n_iter+std_n_iter))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
    for aid = 1:num_alpha % list_alpha_select'
        alpha_id = list_alpha_select(aid);
        p_sign = signrank(list_n_iter_vid{1}(:,alpha_id),list_n_iter_vid{2}(:,alpha_id)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y+step_ns;
        else
            y = list_y;
        end
        text(list_alpha(aid),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center');
    end
    ylabel('Number of iterations');
    xlabel('\alpha');
    set(gca,'FontSize',17);
    set(gca,'XScale','log');
    xlim([10.^floor(log10(min_alpha)-1),10^ceil(log10(max_alpha))]);
    xticks(10.^(floor(log10(min_alpha)-1):ceil(log10(max_alpha))));
    legend({'Floating \alpha','Fixed \alpha'},'Location','NorthEast');
    title([video,' videos'])
    saveas(gcf,sprintf('ABO niter alpha, %s Video, alpha=%s-%s 0910.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO niter alpha, %s Video, alpha=%s-%s 0910.png',video,num2str(min_alpha),num2str(max_alpha)));
end


%% Plot n_iter vs alpha for fixed and floating alpha
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14]+4;
list_p_cutoff = [0.05, 0.005, 0];
grid_score = 0.05;
ratio_step = 1/20;
ratio_ns = 1/5;

% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([6,7],:);
colors = color_many([5,8],:);
max_alpha = 300;
min_alpha = 0.1;
step = 2;
load('include\Time_alpha_ABO.mat','Table_time_all')
load('include\Niter_alpha_ABO.mat','list_alpha_all','list_n_iter_all','list_video','list_addon')

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),500,450],'Color','w');
    hold on;
    list_time_per_iter_vid = cell(length(list_addon),1);
    for addid = 1:length(list_addon)
        list_alpha = list_alpha_all{addid,vid};
        Table_time = Table_time_all{addid,vid};
        Table_time = Table_time(:,1:end-1);
        list_n_iter = list_n_iter_all{addid,vid};
        time_per_iter = Table_time./list_n_iter;
        list_time_per_iter_vid{addid} = time_per_iter;
        list_alpha_select = find((list_alpha<=max_alpha) & (list_alpha>=min_alpha));
        list_alpha_select = list_alpha_select(1:step:end);
        list_alpha = list_alpha(list_alpha_select);
        Table_time = Table_time(:,list_alpha_select);
        list_n_iter = list_n_iter(:,list_alpha_select);
        time_per_iter = time_per_iter(:,list_alpha_select);
        num_alpha = length(list_alpha);
        mean_time_iter = squeeze(mean(time_per_iter,1));
        std_time_iter = squeeze(std(time_per_iter,1,1));
%         [max_F1, pos] = max(mean_F1,[],2);
%         num_alpha = length(list_alpha);

        h=plot(list_alpha',mean_time_iter,'.-','MarkerSize',18,'LineWidth',2,'Color',colors(addid,:)); % 
        errorbar(list_alpha',mean_time_iter,std_time_iter,std_time_iter,'LineWidth',1,'HandleVisibility','off','Color',colors(addid,:)); % 
        yl2 = [0,0.7];
%         yl2(1) = floor(min(min(mean_time_iter-std_time_iter))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_time_iter+std_time_iter))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(mean_time_iter+std_time_iter))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
        ylim(yl2);
    end
    for aid = 1:num_alpha % list_alpha_select'
        alpha_id = list_alpha_select(aid);
        p_sign = signrank(list_time_per_iter_vid{1}(:,alpha_id),list_time_per_iter_vid{2}(:,alpha_id)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y+step_ns;
        else
            y = list_y;
        end
        text(list_alpha(aid),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
            'HorizontalAlignment','Center');
    end
    ylabel('Time per iteration (s)');
    xlabel('\alpha');
    set(gca,'FontSize',17);
    set(gca,'XScale','log');
    xlim([10.^floor(log10(min_alpha)-1),10^ceil(log10(max_alpha))]);
    xticks(10.^(floor(log10(min_alpha)-1):ceil(log10(max_alpha))));
    legend({'Floating \alpha','Fixed \alpha'},'Location','West');
    title([video,' videos'])
    saveas(gcf,sprintf('ABO time_per_iter alpha, %s Video, alpha=%s-%s 0910.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('ABO time_per_iter alpha, %s Video, alpha=%s-%s 0910.png',video,num2str(min_alpha),num2str(max_alpha)));
end


%% Plot correlation vs alpha for fixed and floating alpha
% spike_type = 'simulation';
% simu_opt = '10s_30Hz';
% list_video={'Raw','SNR'};
% % video='SNR'; % {'Raw','SNR'}
% method = 'ours'; % {'FISSA','ours'}
% list_sigma_from = {'Raw','Unmix'}; % {'Unmix'}; % 
% % list_Table_ext = cell(1,1,4);
% colors = distinguishable_colors(17);
% right = 'Residue'; % {'Correlation','Residue','MSE','Pctmin'};
% colors = color([6,7],:);
colors = color_many([5,11,8,14,13],:);
max_alpha = 1000;
min_alpha = 0.1;
step = 1;
% load('NAOMi\corr_split_fix_float_alpha_ksd-psd.mat')
load('NAOMi\corr_split_l1_alpha_ksd-psd.mat')

for vid = 1:length(list_video)
    video = list_video{vid};
    figure('Position',[100,100+450*(vid-1),450,400],'Color','w');
    hold on;
    for addid = 1:length(list_addon)
        addon = list_addon{addid};
        list_alpha = Table_list_alpha{addid,vid};
        recall_CV = Table_recall_CV{addid,vid};
        precision_CV = Table_precision_CV{addid,vid};
        corr_CV = Table_corr_CV{addid,vid};
        thred_ratio_CV = Table_thred_ratio_CV{addid,vid};
        if addid==1
            list_alpha_select = find((list_alpha<100) & (list_alpha>=min_alpha));
        else
            list_alpha_select = find((list_alpha<=max_alpha) & (list_alpha>=min_alpha));
        end
        list_alpha_select = list_alpha_select(1:step:end);
        list_alpha = list_alpha(list_alpha_select);
        list_corr = corr_CV(:,list_alpha_select,:);
        num_alpha = length(list_alpha);
        mean_corr = squeeze(mean(list_corr,1));
        std_corr = squeeze(std(list_corr,1,1));
%         [max_F1, pos] = max(mean_F1,[],2);
%         num_alpha = length(list_alpha);

        h=plot(list_alpha',mean_corr,'.-','MarkerSize',18,'LineWidth',2,'Color',colors(addid,:)); % 
        errorbar(list_alpha',mean_corr,std_corr,std_corr,'LineWidth',1,'HandleVisibility','off','Color',colors(addid,:)); % 
        yl2 = [0.0,1.0];
%         yl2(1) = floor(min(min(mean_F1-std_F1))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(mean_F1+std_F1))/grid_score)*grid_score;
        step_star = (yl2(2)-yl2(1))*ratio_step;
        step_ns = step_star*ratio_ns;
        list_y = max(max(mean_corr+std_corr))+step_star;
%         yl2(2) = ceil(list_y/grid_score)*grid_score;
%         ylim(yl2);
    end
    for aid = 1:num_alpha % list_alpha_select'
        alpha_id = list_alpha_select(aid);
        p_sign = signrank(Table_corr_CV{1,vid}(:,alpha_id,:),Table_corr_CV{end,vid}(:,alpha_id,:)); % 
        p_range = find(p_sign > list_p_cutoff,1,'first');
        if p_range == 1
            y = list_y+step_ns;
        else
            y = list_y;
        end
%         text(list_alpha(aid),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
%             'HorizontalAlignment','Center');
    end
    ylabel('Correlation');
    xlabel('\alpha');
    set(gca,'FontSize',14);
    set(gca,'XScale','log');
    xlim([min_alpha,10^ceil(log10(max_alpha))]);
    xticks(10.^(floor(log10(min_alpha)):ceil(log10(max_alpha))));
%     legend({'Floating \alpha','<=1 sub-trace','Fixed \alpha'},'Location','SouthWest'); % 
    legend({'l_1=0','l_1=0.2','l_1=0.5','l_1=0.8','l_1=1'},'Location','SouthWest','NumColumns',3); % 
    title([video,' videos'])
%     saveas(gcf,sprintf('NAOMi corr float fix alpha split, %s Video, alpha=%s-%s 0809.emf',video,num2str(min_alpha),num2str(max_alpha)));
%     saveas(gcf,sprintf('NAOMi corr float fix alpha split, %s Video, alpha=%s-%s 0809.png',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('NAOMi corr l1 alpha split, %s Video, alpha=%s-%s 0809.emf',video,num2str(min_alpha),num2str(max_alpha)));
    saveas(gcf,sprintf('NAOMi corr l1 alpha split, %s Video, alpha=%s-%s 0809.png',video,num2str(min_alpha),num2str(max_alpha)));
end


%% Compare F1 of our method and FISSA on SNR/raw videos.
spike_type = 'exclude'; % {'1p','exclude','NAOMi'};
% video = 'SNR'; % {'Raw','SNR'}
load([spike_type,'\n_iter FISSA ours.mat'])
xtext = cellfun(@(x) [x,' videos'], list_video, 'UniformOutput',false);

num_video = length(list_video);
num_method = length(list_method);
num_Exp = size(list_F1_all{1},1);
[F1_CV, alpha_CV, list_n_iter_mean] = deal(zeros(num_Exp,num_method,num_video));

for vid = 1:num_video
    video = list_video{vid};
    for mid = 1:length(list_method)
%         method = list_method(mid);
        list_F1 = list_F1_all{mid,vid};
        list_F1_2 = reshape(list_F1,num_Exp,[]);
        [n1,n2,n3] = size(list_F1);
        list_alpha = list_alpha_all{mid,vid};
        for CV = 1:num_Exp
            train = setdiff(1:num_Exp,CV);
            mean_F1 = squeeze(mean(list_F1_2(train,:),1));
            [~,ind_param] = max(mean_F1);
            F1_CV(CV,mid,vid) = list_F1_2(CV,ind_param);
            [ind_alpha,~] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,mid,vid) = alpha;
            
            list_n_iter = list_n_iter_all{CV,mid,vid};
            list_n_iter_mean(CV,mid,vid) = mean(list_n_iter(ind_alpha,:),2);
        end
    end
end
% %%
figure('Position',[100,100,400,400],'Color','w');
data = squeeze(mean(list_n_iter_mean,1))';
err = squeeze(std(list_n_iter_mean,1,1))'; %/sqrt(9)
b=bar(data);       
b(1).FaceColor  = color(2,:);
b(2).FaceColor  = color(5,:);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Number of iterations')
xticklabels(xtext);
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,squeeze(list_n_iter_mean(1:num_Exp,i,:)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
% list_y_line = 0.97+(0:5)*0.02;
% list_y_star = list_y_line+0.005;
list_y_line = 25000;
list_y_star = 26000;
line([xpoints(1,1),xpoints(1,2)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,1),xpoints(2,2)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:)); % 1p
text(xpoints(2,1),list_y_star(1)+500,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:)); % ABO

legend({'FISSA','TUnCaT'},'Interpreter','none',...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
% title(['F1 of ',video,' video'])
box off
% set(gca,'YScale','log');
% ylim([100,1e5]);
ylim([0,30000]);
% set(gca,'Position',two_errorbar_position);
% saveas(gcf,['n_iter_ABO FISSA ours 0713.emf']);
saveas(gcf,['n_iter_',spike_type,' FISSA ours linear 0713.png']);


