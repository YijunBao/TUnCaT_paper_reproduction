color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
addpath('..\plot tools');


%% plot the separation time
video='SNR'; % {'Raw','SNR'}

figure('Position',[100,600,400,500],'Color','w'); 
hold on

max_iter=20000; % {1000,20000}
% load(sprintf('exclude\\Table_time_ours_%s_max_diag11_v1.mat',video),'Table_time','list_alpha');
% load(sprintf('exclude\\Table_time_ours_%s (tol=1e-4, max_iter=%d).mat',video,max_iter),'Table_time','list_alpha');
load(sprintf('exclude\\Table_time_ours_%s_median_diag11_v1.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_ours = total_time(1:end-1); % +total_time(1)
std_time_ours = std_time(1:end-1); % +total_time(1)
list_alpha_ours = list_alpha; % [0.3, 0.5, 1, 2, 3];

load(sprintf('exclude\\Table_time_FISSA_%s (tol=1e-4, max_iter=%d).mat',video,max_iter),'Table_time','list_alpha');
% load(sprintf('Table_time_%s_FISSA.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_FISSA = total_time(1:end-1); % +total_time(1)
std_time_FISSA = std_time(1:end-1); % +total_time(1)
list_alpha_FISSA = list_alpha; % [0.3, 0.5, 1, 2, 3]/10;

% shadedplot(list_alpha_ours, total_time_ours-std_time_ours, total_time_ours+std_time_ours, 1-(1-color(1,:))/2,'w');
% shadedplot(list_alpha_FISSA, total_time_FISSA-std_time_FISSA, total_time_FISSA+std_time_FISSA, 1-(1-color(2,:))/2,'w');
plot(list_alpha_ours, total_time_ours,'color',color(1,:),'LineWidth',2); % ,'o-'
errorbar(list_alpha_ours, total_time_ours,std_time_ours,'color',color(1,:),'HandleVisibility','off')
plot(list_alpha_FISSA, total_time_FISSA,'color',color(2,:),'LineWidth',2); % ,'o-'
errorbar(list_alpha_FISSA, total_time_FISSA,std_time_FISSA,'color',color(2,:),'HandleVisibility','off')

max_iter=1000; % {1000,20000}
% load(sprintf('exclude\\Table_time_ours_%s_max_diag11_v1.mat',video),'Table_time','list_alpha');
load(sprintf('exclude\\Table_time_ours_%s_median_diag11_v1_iter1000.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_ours = total_time(1:end-1); % +total_time(1)
std_time_ours = std_time(1:end-1); % +total_time(1)
list_alpha_ours = list_alpha; % [0.3, 0.5, 1, 2, 3];

load(sprintf('exclude\\Table_time_FISSA_%s (tol=1e-4, max_iter=%d).mat',video,max_iter),'Table_time','list_alpha');
% load(sprintf('Table_time_%s_FISSA.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_FISSA = total_time(1:end-1); % +total_time(1)
std_time_FISSA = std_time(1:end-1); % +total_time(1)
list_alpha_FISSA = list_alpha; % [0.3, 0.5, 1, 2, 3]/10;

% shadedplot(list_alpha_ours, total_time_ours-std_time_ours, total_time_ours+std_time_ours, 1-(1-color(1,:))/2,'w');
% shadedplot(list_alpha_FISSA, total_time_FISSA-std_time_FISSA, total_time_FISSA+std_time_FISSA, 1-(1-color(2,:))/2,'w');
plot(list_alpha_ours, total_time_ours,'--','color',color(1,:),'LineWidth',2); % 
errorbar(list_alpha_ours, total_time_ours,std_time_ours,'color',color(1,:),'HandleVisibility','off')
plot(list_alpha_FISSA, total_time_FISSA,'--','color',color(2,:),'LineWidth',2); % 
errorbar(list_alpha_FISSA, total_time_FISSA,std_time_FISSA,'color',color(2,:),'HandleVisibility','off')

xlabel('\alpha');
ylabel('Unmixing time (s)');
set(gca, 'XScale', 'log');
xlim([0.01,10]);
xticks([0.01,0.1,1,10]);
% ylim([0,80]);
legend({'Our method, \leq20000 iters','FISSA, \leq20000 iters',...
    'Our method, \leq1000 iters','FISSA, \leq1000 iters'},...
    'Location','northoutside','numcolumns',1); % 'Interpreter','tex',
% title(sprintf('%s video',video),'Interpreter','none');
title('Unmixing time');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);
saveas(gcf,sprintf('Speed_methods_iter_%s 0123.png',video));
saveas(gcf,sprintf('Speed_methods_iter_%s 0123.emf',video));


%% plot the separation time
video='SNR'; % {'Raw','SNR'}
max_iter=20000; % {1000,20000}

% load(sprintf('exclude\\Table_time_ours_%s_max_diag11_v1.mat',video),'Table_time','list_alpha');
load(sprintf('exclude\\Table_time_ours_%s (tol=1e-4, max_iter=%d).mat',video,max_iter),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_ours = total_time(1:end-1); % +total_time(1)
std_time_ours = std_time(1:end-1); % +total_time(1)
list_alpha_ours = list_alpha; % [0.3, 0.5, 1, 2, 3];

load(sprintf('exclude\\Table_time_FISSA_%s (tol=1e-4, max_iter=%d).mat',video,max_iter),'Table_time','list_alpha');
% load(sprintf('Table_time_%s_FISSA.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_FISSA = total_time(1:end-1); % +total_time(1)
std_time_FISSA = std_time(1:end-1); % +total_time(1)
list_alpha_FISSA = list_alpha; % [0.3, 0.5, 1, 2, 3]/10;

figure('Position',[100,600,400,400],'Color','w'); 
hold on
% shadedplot(list_alpha_ours, total_time_ours-std_time_ours, total_time_ours+std_time_ours, 1-(1-color(1,:))/2,'w');
% shadedplot(list_alpha_FISSA, total_time_FISSA-std_time_FISSA, total_time_FISSA+std_time_FISSA, 1-(1-color(2,:))/2,'w');
plot(list_alpha_ours, total_time_ours,'color',color(1,:),'LineWidth',2); % ,'o-'
errorbar(list_alpha_ours, total_time_ours,std_time_ours,'color',color(1,:),'HandleVisibility','off')
plot(list_alpha_FISSA, total_time_FISSA,'color',color(2,:),'LineWidth',2); % ,'o-'
errorbar(list_alpha_FISSA, total_time_FISSA,std_time_FISSA,'color',color(2,:),'HandleVisibility','off')

xlabel('\alpha');
ylabel('Unmixing time (s)');
set(gca, 'XScale', 'log');
xlim([0.001,10]);
xticks([0.001,0.01,0.1,1,10]);
% ylim([0,80]);
legend('Our method','FISSA','Location','northoutside','numcolumns',2);
% title(sprintf('%s video',video),'Interpreter','none');
title('Unmixing time');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);
% saveas(gcf,sprintf('Speed_methods_%s_%d 0123.png',video,max_iter));
% saveas(gcf,sprintf('Speed_methods_%s_%d 0123.emf',video,max_iter));


%% plot the separation time
method = 'ours'; % {'FISSA','ours'}
max_iter=20000; % {1000,20000}
video='SNR'; % {'Raw','SNR'}
list_ndiag= {'diag1', 'diag'};
list_vsub= {'v1','v2'};
figure('Position',[100,600,400,400]); 
hold on

for ind_vsub = 1:length(list_vsub)
    vsub = list_vsub{ind_vsub};
    for ind_ndiag = 1:length(list_ndiag)
        ndiag = list_ndiag{ind_ndiag};
        
        load(sprintf('Table_time_%s_%s_%s.mat',video, ndiag, vsub),'Table_time','list_alpha');
        total_time = mean(Table_time,1);
        std_time = std(Table_time,1,1);
        total_time_ours = total_time(1:end-1); % +total_time(1)
        std_time_ours = std_time(1:end-1); % +total_time(1)
        list_alpha_ours = list_alpha; % [0.3, 0.5, 1, 2, 3];

        curve = plot(list_alpha_ours, total_time_ours,'LineWidth',2); % ,'color',color(1,:),'o-'
        errorbar(list_alpha_ours, total_time_ours,std_time_ours ,'color',curve.Color,'HandleVisibility','off'); %
    end
end
        
load(sprintf('Table_time_ours_%s.mat',video),'Table_time','list_alpha');
total_time = mean(Table_time,1);
std_time = std(Table_time,1,1);
total_time_ours = total_time(1:end-1); % +total_time(1)
std_time_ours = std_time(1:end-1); % +total_time(1)
list_alpha_ours = list_alpha; % [0.3, 0.5, 1, 2, 3];

curve = plot(list_alpha_ours, total_time_ours,'LineWidth',2); % ,'color',color(1,:),'o-'
errorbar(list_alpha_ours, total_time_ours,std_time_ours ,'color',curve.Color,'HandleVisibility','off'); %

xlabel('alpha');
ylabel('Unmixing time (s)');
set(gca, 'XScale', 'log');
xlim([0.01,10]);
ylim([0,80]);
legend({'Different background, 1+n+1','Different background, 1+n+2',...
    'Same background, 1+n+1','Same background, 1+n+2','No background subtraction, 1+n+2'});
title(sprintf('%s video',video),'Interpreter','none');
set(gca,'FontSize',11);
saveas(gcf,sprintf('Speed_diag_%s.png',video));


%% Compare processing time of all methods on full ABO videos.
load('time_ABO_full 0910.mat','time','list_method','list_videos');
% select = [1,2,4,3];
select = [2,4,5,3];
list_method = list_method(select);
% load('F1_maxiter_video_method_sigma.mat','F1');
figure('Position',[100,100,400,400],'Color','w');
% x = 1:5;
speed=23000./time(:,[select,select+5]);
speed = permute(reshape(speed,[10,length(select),2]),[1,3,2]);
data = squeeze(mean(speed,1));
err = squeeze(std(speed,1,1)); %/sqrt(9)
b=bar(data);       
b(4).FaceColor  = color(5,:);
% ylim([0.84,0.96]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Speed (frames/s)')
% xlabel('Preprocessing methods')
xticklabels(list_videos);
% title('Accuracy between different methods');

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,speed(1:10,(i-1)*numgroups+(1:numgroups)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None','HandleVisibility','off');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1,'HandleVisibility','off');
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);
hold on; plot([0.5,2.5],[30,30],'k--')
% list_method = [list_method,{'Video rate'}];

xpoints=(1:numgroups)' - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
% list_y_line = 0.97+(0:5)*0.03;
% list_y_star = list_y_line+0.01;
% list_y_line = 490+(0:5)*30;
% list_y_star = list_y_line+6;
list_y_line = 200+(0:5)*15;
list_y_star = list_y_line+3;
line([xpoints(1,1),xpoints(1,4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(1,:));
line([xpoints(1,2),xpoints(1,4)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(2,:));
% text(xpoints(1,2),list_y_star(2)+7,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(2,:));
line([xpoints(1,3),xpoints(1,4)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(1,3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',color(3,:));
% text(xpoints(1,3),list_y_star(3)+5,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',color(3,:));
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
% ylim([0.6,1.05]);
% ylim([0.2,1.2]);

legend(list_method,'Interpreter','none','NumColumns',2,...
    'Location','northoutside','FontName','Arial','FontSize',14);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['time_ABO_full ',num2str(length(list_method)),' 0910.emf']);
saveas(gcf,['time_ABO_full ',num2str(length(list_method)),' 0910.png']);

