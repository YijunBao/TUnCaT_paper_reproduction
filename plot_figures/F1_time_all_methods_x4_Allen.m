color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
green = [0.1,0.9,0.1];
red = [0.9,0.1,0.1];
blue = [0.1,0.8,0.9];
yellow = [0.9,0.9,0.1];
magenta = [0.9,0.3,0.9];
colors = distinguishable_colors(16);

%% Plot the cross validation F1 and processing time for ABO
addon = '_novideounmix_r2_mixout'; % '_pertmin=0.16_eps=0.1_range'; %  
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
spike_type = 'include_BGSubs'; % {'include','exclude','only'};
load(sprintf('%s\\timing_all_methods_split %s%s x5.mat',spike_type,addon,baseline_std),...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all')
num_video = length(list_video);
num_Exp = 10;

order = [2,4,5,3];
list_recall_all = list_recall_all(order,:);
list_precision_all = list_precision_all(order,:);
list_F1_all = list_F1_all(order,:);
list_alpha_all = list_alpha_all(order,:);
list_thred_ratio_all = list_thred_ratio_all(order,:);
list_alpha_all_time = list_alpha_all_time(order,:);
Table_time_all = Table_time_all(order,:);
list_method = list_method(order);
list_method{4} = 'TUnCaT';
list_method{3} = 'Allen SDK';
num_method = length(list_method);

[recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV] = deal(zeros(num_Exp,num_method,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_method,num_video));

for vid = 1:num_video
for mid = 1:num_method
    list_recall = list_recall_all{mid,vid};
    list_precision = list_precision_all{mid,vid};
    list_F1 = list_F1_all{mid,vid};
    Table_time = Table_time_all{mid,vid};
    list_thred_ratio = list_thred_ratio_all{mid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{mid,vid} = zeros(n1,n3);
        Table_precision_CV{mid,vid} = zeros(n1,n3);
        Table_F1_CV{mid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{mid,vid};
        list_alpha_time = list_alpha_all_time{mid,vid};
    else
        Table_recall_CV{mid,vid} = squeeze(list_recall);
        Table_precision_CV{mid,vid} = squeeze(list_precision);
        Table_F1_CV{mid,vid} = squeeze(list_F1);
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
        recall_CV(CV,mid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,mid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,mid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,mid,vid) = alpha;
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_thred_ratio);
            Table_recall_CV{mid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{mid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{mid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,mid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
        else
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,mid,vid) = Table_time(CV);
        end
    end
end
end

save(['F1_time_ABO',spike_type,addon,baseline_std,' x4.mat'],'list_method','list_video','recall_CV',...
    'precision_CV','F1_CV','time_CV','thred_ratio_CV','alpha_CV',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');

% data = squeeze(mean(permute(thred_ratio_CV(:,1:end,:),[1,3,2]),1))';
% data = squeeze(std(permute(thred_ratio_CV(:,1:end,:),[1,3,2]),1,1))';
%% plot F1 of all methods 
load(['F1_time_ABO',spike_type,addon,baseline_std,' x4.mat'],'list_method','list_video','F1_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(F1_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[100,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('{\itF}_1')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
% list_y_line = 0.97+(0:5)*0.03;
% list_y_star = list_y_line+0.01;
list_y_line = 1+(0:5)*0.04;
list_y_star = list_y_line+0.01;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(2),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(2,:));

line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(4),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(4,:));
ylim([0.5,1.1]);
yticks(0.5:0.1:1);
% ylim([0.2,1.2]);

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','northoutside','FontName','Arial'); % ,'FontSize',18
box off
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('F1 split x%d ABO%s%s 0908.png',num_method,addon,baseline_std));
saveas(gcf,sprintf('F1 split x%d ABO%s%s 0908.emf',num_method,addon,baseline_std));


%% plot processing time of all methods 
load(['F1_time_ABO',spike_type,addon,baseline_std,' x4.mat'],'list_method','list_video','F1_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(time_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[500,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
% ylim([0.84,0.96]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Processing time (s)')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 3000*1.5.^(0:5);
list_y_star = list_y_line*1.1;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% text(xpoints(1,3),list_y_star(3)*1.2,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(2),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(2,:));

% list_y_line = list_y_line;
% list_y_star = list_y_star;
line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(4),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(4,:));

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','NorthOutside','FontName','Arial'); % ,'FontSize',18
box off
set(gca,'YScale','log');
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('time split x%d ABO%s%s 0908.png',num_method,addon,baseline_std));
saveas(gcf,sprintf('time split x%d ABO%s%s 0908.emf',num_method,addon,baseline_std));

%% plot ROC curve of all methods 
load(['F1_time_ABO',spike_type,addon,baseline_std,' x4.mat'],'list_method','list_video',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');
[num_method,num_video] = size(Table_F1_CV);

figure('Position',[100,550,900,400]); 
for vid = 1:num_video
subplot(1,num_video,vid);
hold on
for mid = 1:num_method
    h = plot(mean(Table_recall_CV{mid,vid},1), mean(Table_precision_CV{mid,vid},1),'LineWidth',2); % ,'o-','color',color(1,:)
    if mid==num_method
        h.Color = color(5,:);
    end
end

xlabel('Recall');
ylabel('Precision');
% xlim([0,1]);
% ylim([0,1]);
box off
title([list_video{vid},' video']);
set(gca,'FontSize',14);
end
subplot(1,num_video,1);
legend(list_method,'Interpreter','none','numcolumns',1,...
    'Location','SouthWest','FontName','Arial','FontSize',14);
% saveas(gcf,sprintf('F1 %s %d 0503.emf',simu_opt,numbars));
saveas(gcf,sprintf('ROC split x%d ABO%s%s 0908.png',num_method,addon,baseline_std));



%% Plot the cross validation F1 and processing time for 1p
addon = '_merge_novideounmix_r2_mixout'; % '_pertmin=0.16_eps=0.1_range'; %  
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
load(['1p\timing_all_methods_split ',addon,baseline_std,' x5.mat'],...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all','list_thred_ratio_all')
num_video = length(list_video);
num_Exp = 9;

order = [2,4,5,3];
list_recall_all = list_recall_all(order,:);
list_precision_all = list_precision_all(order,:);
list_F1_all = list_F1_all(order,:);
list_alpha_all = list_alpha_all(order,:);
list_thred_ratio_all = list_thred_ratio_all(order,:);
list_alpha_all_time = list_alpha_all_time(order,:);
Table_time_all = Table_time_all(order,:);
list_method = list_method(order);
list_method{4} = 'TUnCaT';
list_method{3} = 'Allen SDK';
num_method = length(list_method);

[recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV] = deal(zeros(num_Exp,num_method,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_method,num_video));

for vid = 1:num_video
for mid = 1:num_method
    list_recall = list_recall_all{mid,vid};
    list_precision = list_precision_all{mid,vid};
    list_F1 = list_F1_all{mid,vid};
    Table_time = Table_time_all{mid,vid};
    list_thred_ratio = list_thred_ratio_all{mid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{mid,vid} = zeros(n1,n3);
        Table_precision_CV{mid,vid} = zeros(n1,n3);
        Table_F1_CV{mid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{mid,vid};
        list_alpha_time = list_alpha_all_time{mid,vid};
    else
        Table_recall_CV{mid,vid} = squeeze(list_recall);
        Table_precision_CV{mid,vid} = squeeze(list_precision);
        Table_F1_CV{mid,vid} = squeeze(list_F1);
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [val,ind_param] = max(mean_F1);
        recall_CV(CV,mid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,mid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,mid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,mid,vid) = alpha;
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_thred_ratio);
            Table_recall_CV{mid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{mid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{mid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,mid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
        else
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,mid,vid) = Table_time(CV);
        end
    end
end
end
F1_CV(isnan(F1_CV))=0;

save('F1_time_1p x4.mat','list_method','list_video','recall_CV',...
    'precision_CV','F1_CV','time_CV','thred_ratio_CV','alpha_CV',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');

%% plot F1 of all methods 
load('F1_time_1p x4.mat','list_method','list_video','F1_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(F1_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[100,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('{\itF}_1')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:num_Exp,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.93+(0:5)*0.08;
list_y_star = list_y_line+0.02;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,1),list_y_star(1),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
text(xpoints(1,1),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
% text(xpoints(1,2),list_y_star(2)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(2),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(2,:));

line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(4),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(4,:));
% ylim([0.5,1.0]);

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','northoutside','FontName','Arial','FontSize',14);
box off
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('F1 split x%d 1p %s%s 0908.png',num_method,addon,baseline_std));
saveas(gcf,sprintf('F1 split x%d 1p %s%s 0908.emf',num_method,addon,baseline_std));


%% plot processing time of all methods 
load('F1_time_1p x4.mat','list_method','list_video','time_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(time_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[500,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
% ylim([0.84,0.96]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Processing time (s)')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:num_Exp,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 400*1.5.^(0:5);
list_y_star = list_y_line*1.1;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
text(xpoints(1,2),list_y_star(2)*1.2,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(3),'*','HorizontalAlignment', 'right','FontSize',14,'Color',color(4,:));
text(xpoints(1,3),list_y_star(3)*1.2,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4)*1.3,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(4,:));

% list_y_line = list_y_line-21;
% list_y_star = list_y_star-21;
line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% text(xpoints(4),list_y_star(1)+0.018,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',color(4,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','NorthOutside','FontName','Arial','FontSize',14);
box off
set(gca,'YScale','log');
ylim([1,1000]);
yticks(10.^(-1:4));
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('time split x%d 1p %s%s 0908.png',num_method,addon,baseline_std));
saveas(gcf,sprintf('time split x%d 1p %s%s 0908.emf',num_method,addon,baseline_std));

%% plot ROC curve of all methods 
load('F1_time_1p x4.mat','list_method','list_video',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');
[num_method,num_video] = size(Table_F1_CV);

figure('Position',[100,500,900,400]); 
for vid = 1:num_video
subplot(1,num_video,vid);
hold on
for mid = 1:num_method
    h = plot(mean(Table_recall_CV{mid,vid},1), mean(Table_precision_CV{mid,vid},1),'LineWidth',2); % ,'o-','color',color(1,:)
    if mid==num_method
        h.Color = color(5,:);
    end
end

xlabel('Recall');
ylabel('Precision');
% xlim([0,1]);
% ylim([0,1]);
box off
title([list_video{vid},' video']);
set(gca,'FontSize',14);
end
subplot(1,num_video,1);
legend(list_method,'Interpreter','none','numcolumns',1,...
    'Location','SouthWest','FontName','Arial','FontSize',14);
% saveas(gcf,sprintf('F1 %s %d 0503.emf',simu_opt,numbars));
saveas(gcf,sprintf('ROC split x%d 1p %s%s 0908.png',num_method,addon,baseline_std));



% data = squeeze(mean(permute(F1_CV(:,1:end,:),[1,3,2]),1))';
%% Plot the cross validation F1 and processing time for simulated videos
% simu_opt = '100s_30Hz_100+10_rand_vary-rate'; 
simu_opt = '120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f'; %   thred_ratio = 1-6
% simu_opt = '1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c'; %   thred_ratio = 1-6
addon = '_novideounmix_r2_mixout'; %_corr_fixed_alpha '_pertmin=0.16_eps=0.1_range'; % 
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
load(['NAOMi\timing_',simu_opt,'_all_methods_split ',addon,baseline_std,' x5.mat'],...
    'list_alpha_all_time','Table_time_all','list_method','list_video',...
    'list_recall_all','list_precision_all','list_F1_all','list_alpha_all',...
    'list_thred_ratio_all','list_corr_unmix_all')
num_video = length(list_video);
num_Exp = 10;

order = [2,4,5,3];
list_recall_all = list_recall_all(order,:);
list_precision_all = list_precision_all(order,:);
list_F1_all = list_F1_all(order,:);
list_alpha_all = list_alpha_all(order,:);
list_thred_ratio_all = list_thred_ratio_all(order,:);
list_alpha_all_time = list_alpha_all_time(order,:);
list_corr_unmix_all = list_corr_unmix_all(order,:);
Table_time_all = Table_time_all(order,:);
list_method = list_method(order);
list_method{4} = 'TUnCaT';
list_method{3} = 'Allen SDK';
num_method = length(list_method);

[recall_CV, precision_CV, F1_CV, corr_CV, time_CV, alpha_CV, thred_ratio_CV, alpha_corr_CV, time_corr_CV] = deal(zeros(num_Exp,num_method,num_video));
[Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_method,num_video));

for vid = 1:num_video
for mid = 1:num_method
    list_recall = list_recall_all{mid,vid};
    list_precision = list_precision_all{mid,vid};
    list_F1 = list_F1_all{mid,vid};
    list_corr_unmix = list_corr_unmix_all{mid,vid};
    list_corr_unmix = cellfun(@mean, list_corr_unmix);
    Table_time = Table_time_all{mid,vid};
    list_thred_ratio = list_thred_ratio_all{mid,vid};
    list_recall_2 = reshape(list_recall,num_Exp,[]);
    list_precision_2 = reshape(list_precision,num_Exp,[]);
    list_F1_2 = reshape(list_F1,num_Exp,[]);
    [n1,n2,n3] = size(list_F1);
    if min(n2,n3)>1
        Table_recall_CV{mid,vid} = zeros(n1,n3);
        Table_precision_CV{mid,vid} = zeros(n1,n3);
        Table_F1_CV{mid,vid} = zeros(n1,n3);
        list_alpha = list_alpha_all{mid,vid};
        list_alpha_time = list_alpha_all_time{mid,vid};
    else
        Table_recall_CV{mid,vid} = list_recall;
        Table_precision_CV{mid,vid} = list_precision;
        Table_F1_CV{mid,vid} = list_F1;
    end
    for CV = 1:num_Exp
        train = setdiff(1:num_Exp,CV);
        mean_F1 = squeeze(mean(list_F1_2(train,:),1));
        [~,ind_param] = max(mean_F1);
        recall_CV(CV,mid,vid) = list_recall_2(CV,ind_param);
        precision_CV(CV,mid,vid) = list_precision_2(CV,ind_param);
        F1_CV(CV,mid,vid) = list_F1_2(CV,ind_param);
        if min(n2,n3)>1
            [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
            alpha = list_alpha(ind_alpha);
            alpha_CV(CV,mid,vid) = alpha;
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_thred_ratio);
            Table_recall_CV{mid,vid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
            Table_precision_CV{mid,vid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
            Table_F1_CV{mid,vid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
            ind_alpha = find(list_alpha_time==alpha);
            time_CV(CV,mid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);

            mean_corr = squeeze(mean(list_corr_unmix(train,:),1));
            [~,ind_corr] = max(mean_corr);
            corr_CV(CV,mid,vid) = list_corr_unmix(CV,ind_corr);
            alpha = list_alpha(ind_corr);
            alpha_corr_CV(CV,mid,vid) = alpha;
            ind_alpha = find(list_alpha_time==alpha);
            time_corr_CV(CV,mid,vid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
        else
            thred_ratio_CV(CV,mid,vid) = list_thred_ratio(ind_param);
            if ~isvector(Table_time)
                Table_time = diag(Table_time);
            end
            time_CV(CV,mid,vid) = Table_time(CV);
            corr_CV(CV,mid,vid) = list_corr_unmix(CV);
            time_corr_CV(CV,mid,vid) = Table_time(CV);
        end
    end
end
end

save(['F1_time_',simu_opt,addon,baseline_std,' x4.mat'],'list_method','list_video','recall_CV',...
    'precision_CV','F1_CV','corr_CV','time_CV','thred_ratio_CV','alpha_CV',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');

%% plot F1 of all methods 
% data = squeeze(mean(permute(F1_CV(:,1:end,:),[1,3,2]),1))';
% data = squeeze(std(permute(F1_CV(:,1:end,:),[1,3,2]),1,1))';
load(['F1_time_',simu_opt,addon,baseline_std,' x4.mat'],'list_method','list_video','F1_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(F1_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));
% disp(data(:,3)');

figure('Position',[100,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
% ylim([0.5,1]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('{\itF}_1')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 1+(0:5)*0.07;
list_y_star = list_y_line+0.015;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
% text(xpoints(1,1),list_y_star(1)+0.025,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
% text(xpoints(1,2),list_y_star(2)+0.025,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,3),list_y_star(3),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% text(xpoints(1,3),list_y_star(3)+0.025,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));

line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,2),list_y_star(2)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:));
text(xpoints(2,2),list_y_star(2),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% ylim([0.2,1.2]);

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','northoutside','FontName','Arial','FontSize',14);
box off
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('F1 split x%d %s %s%s 0908.png',num_method,simu_opt,addon,baseline_std));
saveas(gcf,sprintf('F1 split x%d %s %s%s 0908.emf',num_method,simu_opt,addon,baseline_std));

%% plot correlation of all methods 
load(['F1_time_',simu_opt,addon,baseline_std,' x4.mat'],'list_method','list_video','corr_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(corr_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[500,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
% ylim([0.5,1]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Correlation')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 1.02+(0:5)*0.07;
list_y_star = list_y_line+0.015;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,1),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(1,:));
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,3),list_y_star(3)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(4,:));
% text(xpoints(1,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));

line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
text(xpoints(2,1),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,4),list_y_star(4)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(4,:));
% text(xpoints(2,4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% ylim([0.2,1.2]);

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','northoutside','FontName','Arial','FontSize',14);
box off
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('correlation split x%d %s %s%s 0908.png',num_method,simu_opt,addon,baseline_std));
saveas(gcf,sprintf('correlation split x%d %s %s%s 0908.emf',num_method,simu_opt,addon,baseline_std));

%% plot time of all methods 
load(['F1_time_',simu_opt,addon,baseline_std,' x4.mat'],'list_method','list_video','time_CV');
% num_method = length(list_method);
% num_video = length(list_video);
% data_all = F1_CV;
data_all = permute(time_CV(:,1:end,:),[1,3,2]);
data = squeeze(mean(data_all,1));
err = squeeze(std(data_all,1,1));

figure('Position',[900,100,400,400],'Color','w');
b=bar(data);       
b(end).FaceColor = color(5,:);
% ylim([0.84,0.96]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Processing time (s)')
xticklabels(cellfun(@(x) [x,' video'],list_video, 'UniformOutput', false));
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,data_all(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), min(data(:,i)-0.1,err(:,i)), err(:,i),  'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 240*1.6.^(0:5);
list_y_star = list_y_line*1.1;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
text(xpoints(1,3),list_y_star(3)*1.3,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
% line([xpoints(1,4),xpoints(1,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(1,4),list_y_star(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',color(4,:));
% text(xpoints(1,4),list_y_star(4)*1.3,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(4,:));

% list_y_line = list_y_line-15;
% list_y_star = list_y_star-15;
line([xpoints(2,1),xpoints(2,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(2,2),xpoints(2,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
line([xpoints(2,3),xpoints(2,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
text(xpoints(2,3),list_y_star(3)*1.3,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
% line([xpoints(2,4),xpoints(2,4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2,5),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',color(5,:));
% text(xpoints(2,4),list_y_star(4)*1.3,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(4,:));

legend(list_method(1:numbars),'Interpreter','none','numcolumns',2,...
    'Location','northoutside','FontName','Arial','FontSize',14);
box off
set(gca,'YScale','log');
ylim([0.1,1000]);
% yticks(10.^(-1:4));
% set(gca,'Position',two_errorbar_position);
% title('F1 of all methods, ABO')
saveas(gcf,sprintf('time split x%d %s %s%s 0908.png',num_method,simu_opt,addon,baseline_std));
saveas(gcf,sprintf('time split x%d %s %s%s 0908.emf',num_method,simu_opt,addon,baseline_std));

%% plot ROC curve of all methods 
load(['F1_time_',simu_opt,addon,baseline_std,' x4.mat'],'list_method','list_video','F1_CV',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');
[num_method,num_video] = size(Table_F1_CV);

figure('Position',[100,500,900,400]); 
for vid = 1:num_video
subplot(1,num_video,vid);
hold on
for mid = 1:num_method
    h = plot(mean(Table_recall_CV{mid,vid},1), mean(Table_precision_CV{mid,vid},1),'LineWidth',2); % ,'o-','color',color(1,:)
    if mid==num_method
        h.Color = color(5,:);
    end
end

xlabel('Recall');
ylabel('Precision');
% xlim([0,1]);
% ylim([0,1]);
box off
title([list_video{vid},' video']);
set(gca,'FontSize',14);
end
subplot(1,num_video,2);
legend(list_method,'Interpreter','none','numcolumns',1,...
    'Location','SouthWest','FontName','Arial','FontSize',14);
% saveas(gcf,sprintf('F1 %s %d 0503.emf',simu_opt,numbars));
saveas(gcf,sprintf('ROC split x%d %s %s%s 0908.png',num_method,simu_opt,addon,baseline_std));

