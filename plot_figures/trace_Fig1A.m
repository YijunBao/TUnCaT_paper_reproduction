clear;
addpath(genpath('..\evaluation'));
magenta = [0.9,0.3,0.9];
red = [0.8,0.0,0.0]; 
green = [0.0,0.65,0.0];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
k=10;
Exp_ID=list_Exp_ID{k};
nn=51;

% dir_video='D:\ABO\20 percent 200';
% dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
dir_video='..\data\ABO\';
% dir_traces=dir_video;
dir_traces='..\results\ABO\unmixed traces\';
dir_label = [dir_video,'GT transients\'];
load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
% load(['manual labels\output_',Exp_ID,'.mat'],'output');
% load(['raw\',Exp_ID,'.mat'],'traces','bgtraces');
addon = ''; % '_eps=0.1'; % 
video = 'Raw';
folder = sprintf('traces_ours_%s%s',video,addon);
dir_FISSA = fullfile(dir_traces,folder);
load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
% traces = traces';
% bgtraces_median = bgtraces';
trace = traces(:,nn);
spike_frames = [7934, 8124, 8225, 10422]; 
list_colors = [red; red; red; green];
list_index = {'(i)','(ii)','(iii)','(iv)'};

%%
[mu, sigma] = SNR_normalization(traces,'quantile-based std','median');
thred_ratio=2.5;
thred = mu+thred_ratio*sigma;
[recall, precision, F1,individual_recall,...
    individual_precision,spikes_GT_array,spikes_eval_array]...
    = GetPerformance_SpikeDetection_split(output,traces',thred_ratio,sigma,mu);
spikes_eval = spikes_eval_array{nn};

figure('Position',[50,50,800,300],'Color','w');
plot(trace);
bottom = min(trace)-10;
hold on;
% plot(spike_frames, trace(spike_frames), 'r.','MarkerSize',15);
% plot([1,length(trace)],thred(nn)*ones(1,2),'k--');
for ii = 1:length(spike_frames)
    frame = spike_frames(ii);
    ind = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
    plot((spikes_eval(ind,1):spikes_eval(ind,2)),trace(spikes_eval(ind,1):spikes_eval(ind,2)),'Color',list_colors(ii,:),'LineWidth',2); 
    text(frame, bottom, list_index{ii},'Color',list_colors(ii,:),'HorizontalAlignment','center','FontSize',14);
end
xlim([7500,10500])
ylim([300,600])
set(gca,'FontSize',12);
xticks({});
% yticks({});

pos1=10000;
pos2=530;
plot(pos1+[0,300],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+50*[0,1],'k','LineWidth',2);
text(pos1-160,pos2+25,{'\Delta{\itF}','50'},'FontSize',14,'rotation',0);
text(pos1,pos2-15,'10 s','FontSize',14,'rotation',0);

saveas(gcf,['Neuron ',num2str(nn),', Video ',Exp_ID,' parts.png']);
% saveas(gcf,['Neuron ',num2str(nn),', Video ',Exp_ID,' parts.emf']);
