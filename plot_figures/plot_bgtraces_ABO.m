magenta = [0.9,0.3,0.9];
red = [0.8,0.0,0.0]; 
green = [0.0,0.65,0.0];
list_index = {'(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)'};
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
k=10;
Exp_ID=list_Exp_ID{k};
dir_video='D:\ABO\20 percent 200';
dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
% load(['manual labels\output_',Exp_ID,'.mat'],'output');
% load('example bgtraces.mat','traces','bgtraces_mean','bgtraces_median');
addon = '_novideounmix_r2'; % '_eps=0.1'; % 
video = 'Raw';
folder = sprintf('traces_ours_%s%s',video,addon);
dir_FISSA = fullfile(dir_video,folder);
load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
traces = traces';
bgtraces_median = bgtraces';
nn=16; % 2
spike_frames = [0313,1004,1793,2106];
list_colors = [green; red; red; green];
gap = 0.06;  marg_h = 0.07; marg_w = 0.12; 
% duration = 1500:3999;
duration = 10001:12500;
position = [30,30,800,600];

%% v2
[mu, sigma] = SNR_normalization(traces,'quantile-based std','median');
thred_ratio=2.5;
thred = mu+thred_ratio*sigma;
[recall, precision, F1,individual_recall,...
    individual_precision,spikes_GT_array,spikes_eval_array]...
    = GetPerformance_SpikeDetection(output,traces,thred_ratio,sigma,mu);
spikes_eval = spikes_eval_array{nn}-duration(1)+1;

figure('Position',position,'Color','w');
ha = tight_subplot(3, 1, gap, marg_h, marg_w);
hold on;
list_title = {'Raw trace','Background trace','Background subtracted trace'};
raw_trace = traces(nn,duration);
bg_trace = bgtraces_median(nn,duration);
list_traces = {raw_trace, bg_trace, raw_trace - bg_trace};
% list_traces_all = {traces, bgtraces_median, traces - bgtraces_median};
list_ylim = [300,600; 250,550; -50,250];
for ind=1:3
    axes(ha(ind));
    plot(list_traces{ind},'LineWidth',1) % ,'Color',color(ind,:)
    hold on;
    ylim(list_ylim(ind,:));
%     text(1,2.5,['Actual trace $',char('A'-1+ind),'$'],'Interpreter','latex','FontSize',18,'Color',color(ind,:))
    xticklabels({});
    ylabel('Intensity');
    box off
    title(list_title{ind},'FontWeight','Normal') % ,'FontSize',12
    set(gca,'FontSize',12)

    bottom = min(list_traces{ind})-15;
    for ii = 1:length(spike_frames)
        frame = spike_frames(ii);
        t = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
        if ~isempty(t)
            plot((spikes_eval(t,1):spikes_eval(t,2)),...
                list_traces{ind}(spikes_eval(t,1):spikes_eval(t,2)),'Color',list_colors(ii,:),'LineWidth',2); 
            text(frame, bottom, list_index{ii},'Color',list_colors(ii,:),'HorizontalAlignment','center','FontSize',14);
        end
    end
end

pos1=150;
pos2=150;
plot(pos1+[0,150],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+100*[0,1],'k','LineWidth',2);
text(pos1-20,pos2+50,{'\Delta{\itF}','100'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1,pos2-20,'5 s','FontSize',14,'rotation',0);

% linkaxes(axes);
saveas(gcf,['example background traces median ',num2str(nn),' 0805.emf']);
saveas(gcf,['example background traces median ',num2str(nn),' 0805.png']);

%% v1
% figure('Position',position,'Color','w');
% ha = tight_subplot(3, 1, gap, marg_h, marg_w);
% hold on;
% list_title = {'Raw trace','Background trace','Background subtracted trace'};
% raw_trace = traces(nn,duration);
% bg_trace = bgtraces_median(nn,duration);
% list_traces = {raw_trace, bg_trace, raw_trace - bg_trace};
% list_traces_all = {traces, bgtraces_median, traces - bgtraces_median};
% list_ylim = [300,600; 250,550; -50,250];
% for ind=1:3
%     axes(ha(ind));
%     plot(list_traces{ind},'LineWidth',1) % ,'Color',color(ind,:)
%     hold on;
%     ylim(list_ylim(ind,:));
% %     text(1,2.5,['Actual trace $',char('A'-1+ind),'$'],'Interpreter','latex','FontSize',18,'Color',color(ind,:))
%     xticklabels({});
%     ylabel('Intensity');
%     box off
%     title(list_title{ind},'FontSize',12)
%     set(gca,'FontSize',12)
% 
%     [mu, sigma] = SNR_normalization(list_traces_all{ind},'quantile-based std','median');
%     thred_ratio=2.5;
%     thred = mu+thred_ratio*sigma;
%     [recall, precision, F1,individual_recall,...
%         individual_precision,spikes_GT_array,spikes_eval_array]...
%         = GetPerformance_SpikeDetection(output,list_traces_all{ind},thred_ratio,sigma,mu);
%     spikes_eval = spikes_eval_array{nn}-duration(1)+1;
%     bottom = min(list_traces{ind})-15;
%     for ii = 1:length(spike_frames)
%         frame = spike_frames(ii);
%         t = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
%         if ~isempty(t)
%             plot((spikes_eval(t,1):spikes_eval(t,2)),...
%                 list_traces{ind}(spikes_eval(t,1):spikes_eval(t,2)),'Color','r','LineWidth',2); 
%             text(frame, bottom, list_index{ii},'Color','r','HorizontalAlignment','center','FontSize',14);
%         end
%     end
% end

% linkaxes(axes);
% saveas(gcf,['example background traces median ',num2str(nn),' v1.emf']);
% saveas(gcf,['example background traces median ',num2str(nn),' v1.png']);




%%
figure('Position',position,'Color','w');
ha = tight_subplot(3, 1, gap, marg_h, marg_w);
hold on;
list_title = {'Raw trace','Median background trace','Raw - median'};
raw_trace = traces(nn,duration);
bg_trace = bgtraces_median(nn,duration);
list_traces = {raw_trace, bg_trace, raw_trace - bg_trace};
list_ylim = [300,600; 250,550; -50,250];
for ind=1:3
    axes(ha(ind));
    plot(list_traces{ind},'LineWidth',1) % ,'Color',color(ind,:)
    ylim(list_ylim(ind,:));
%     text(1,2.5,['Actual trace $',char('A'-1+ind),'$'],'Interpreter','latex','FontSize',18,'Color',color(ind,:))
    xticklabels({});
    ylabel('Intensity');
    box off
    title(list_title{ind},'FontSize',12)
    set(gca,'FontSize',12)
end
% linkaxes(axes);
% saveas(gcf,['example background traces median.emf']);
% saveas(gcf,['example background traces median.png']);


%%
figure('Position',position+[0,position(4)+20,0,0],'Color','w');
ha = tight_subplot(3, 1, gap, marg_h, marg_w);
hold on;
list_title = {'Raw trace','Mean background trace','Raw - mean'};
raw_trace = traces(nn,duration);
bg_trace = bgtraces_mean(nn,duration);
list_traces = {raw_trace, bg_trace, raw_trace - bg_trace};
list_ylim = [300,600; 250,550; -50,250];
for ind=1:3
    axes(ha(ind));
    plot(list_traces{ind},'LineWidth',1) % ,'Color',color(ind,:)
    ylim(list_ylim(ind,:));
%     text(1,2.5,['Actual trace $',char('A'-1+ind),'$'],'Interpreter','latex','FontSize',18,'Color',color(ind,:))
    xticklabels({});
    ylabel('Intensity');
    box off
    title(list_title{ind},'FontSize',12)
    set(gca,'FontSize',12)
end
% saveas(gcf,'example background traces 1mean 0122.emf');
% saveas(gcf,'example background traces 1mean 0122.png');


%%
figure('Position',position+[position(3)+20,position(4)+20,0,0],'Color','w');
ha = tight_subplot(3, 1, gap, marg_h, marg_w);
hold on;
list_title = {'Raw trace','Mean background trace','Raw - 0.7\timesmean'};
raw_trace = traces(nn,duration);
bg_trace = bgtraces_mean(nn,duration);
list_traces = {raw_trace, bg_trace, raw_trace - 0.7*bg_trace};
list_ylim = [300,600; 250,550; 100,400];
for ind=1:3
    axes(ha(ind));
    plot(list_traces{ind},'LineWidth',1) % ,'Color',color(ind,:)
    ylim(list_ylim(ind,:));
%     text(1,2.5,['Actual trace $',char('A'-1+ind),'$'],'Interpreter','latex','FontSize',18,'Color',color(ind,:))
    xticklabels({});
    ylabel('Intensity');
    box off
    title(list_title{ind},'FontSize',12)
    set(gca,'FontSize',12)
end
% saveas(gcf,'example background traces 0.7mean 0122.emf');
% saveas(gcf,'example background traces 0.7mean 0122.png');