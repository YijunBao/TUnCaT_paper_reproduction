clear;
addpath(genpath('..\evaluation'));
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% addpath('..\plot tools');


%% Compare processing time of all methods on full ABO videos.
dir_eval = '..\results\ABO\evaluation\';
load([dir_eval,'time_ABO_full 0910.mat'],'time','list_method','list_videos');
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
% saveas(gcf,['S4 Fig.emf']);
saveas(gcf,['S4 Fig.png']);

