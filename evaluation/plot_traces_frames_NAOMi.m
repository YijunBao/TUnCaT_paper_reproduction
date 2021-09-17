magenta = [0.9,0.3,0.9];
red = [0.8,0.0,0.0]; 
green = [0.0,0.65,0.0];
list_index = {'(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)'};

std_method = 'quantile-based std comp';
baseline_method = 'median';
% std_method = 'psd';
% baseline_method = 'ksd';

%%
list_video = {'Raw'}; % 'SNR',
num_video = length(list_video);
% list_method = {'Clean trace'; 'BG subtraction'; 'FISSA'; 'Our unmixing'; 'CNMF'}; % 'Remove overlap'; 'Percent pixels'; ; 'AllenSDK'
list_method = {'Ground truth'; 'BG subtraction'; 'FISSA'; 'CNMF'; 'AllenSDK'; 'TUnCaT'};
num_method = length(list_method);

nn=1;
spike_frames = [134, 1656, 2127, 2239]; % 1 of 8
% nn=32;
% spike_frames = [37, 229, 968, 1573, 2574, 2775]; % 1 of 9
% for nn=1:length(hasGT)
nno=hasGT(nn);
figure('Position',[20,50,900,900]);
[ax, pos] = tight_subplot(num_method,num_video,[0.03,0.04]);
% hold on; 
% clear ax;

win_filt = length(dFF);
T = size(clean_traces,2);
Tf=T-win_filt+1;
range=1:Tf;
range_full = win_filt:T;
clear mu sigma
% sigma from raw traces
% [mu(1), sigma(1)] = SNR_normalization(list_traces_raw_filt{1}(nno,:),std_method);
% [mu(2), sigma(2)] = SNR_normalization(list_traces_raw_filt{2}(nno,:),std_method);

% name_video = repmat({'raw video'},[num_method,1]); % 'SNR video',
% % name_video = repmat({'raw video',''},[num_method,1]);
% name_method = repmat(list_method,[1,2]);
% list_title = cellfun(@(x,y) [x,' for ', y], name_method, name_video, 'UniformOutput', false);
list_title = list_method;

for vid=1:num_video
    spikes_frames = list_spike_frames{vid}{nno};
    for mid = 1:num_method
        if mid == 1
            trace_unmix = list_traces_clean{vid}(nno,:);
            trace_unmix = trace_unmix/min(trace_unmix)-1;
            axes(ax(vid));
            plot(trace_unmix); 
%             spikes_GT=spikes_GT_array{end,vid}{nn};
% 
%             trace_mix=list_traces_raw_filt{vid}(nno,range);
%             trace_mix=(trace_mix-median(trace_mix))./sigma(vid);
% 
%             axes(ax(vid));
%             plot(range,trace_mix); 
%             hold on; 
%             for ii=1:size(spikes_GT,1)
%                 plot(range(spikes_GT(ii,1):spikes_GT(ii,2)),trace_mix(spikes_GT(ii,1):spikes_GT(ii,2)),'Color',magenta,'LineWidth',2); 
%             end
%             ylabel('SNR');
            title(list_title{mid,vid},'FontWeight','Normal');

        else
%             aid = list_ind_alpha(mid-1);
%             alpha = list_alpha(aid);
            spikes_eval=spikes_eval_array{mid-1,vid}{nn};
            spikes_eval(:,1) = max(1,spikes_eval(:,1));
            trace_unmix=list_traces_unmixed{mid-1,vid}(nno,:);
            [mu_unmix, sigma_unmix] = SNR_normalization(trace_unmix,std_method,baseline_method);
%             trace_unmix=(trace_unmix-mu_unmix)./sigma(vid);
            trace_unmix=(trace_unmix-mu_unmix)./sigma_unmix;

            axes(ax(num_video*(mid-1)+vid));
            plot(trace_unmix); 
%             histogram(trace_unmix);
            hold on; 
            for ii=1:size(spikes_eval,1)
                if spikes_eval(ii,3)
                    spike_color = green;
                else
                    spike_color = red;
                end
                transient_range = spikes_eval(ii,1):spikes_eval(ii,2);
                if ~contains(list_video{vid},'SNR')
                    transient_range = transient_range + lag;
                end
                if spikes_eval(ii,1)==spikes_eval(ii,2)
                    plot(range(transient_range),...
                        trace_unmix(transient_range),'.','Color',spike_color,'LineWidth',2); 
                else
                    plot(range(transient_range),...
                        trace_unmix(transient_range),'Color',spike_color,'LineWidth',2);
                end
%                 plot((spikes_eval(ii,1):spikes_eval(ii,2)),trace_unmix(spikes_eval(ii,1):spikes_eval(ii,2)),'Color',red,'LineWidth',2); 
            end
            
            for ii = 1:length(spike_frames)
                frame = spike_frames(ii);
                ind = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
                if ~isempty(ind)
                    if spikes_eval(ind,3)
                        spike_color = green;
                    else
                        spike_color = red;
                    end
%                     if find(abs(spikes_frames-frame)<cons)
%                         spike_color = green;
%                     else
%                         spike_color = red;
%                     end
%                     plot((spikes_eval(ind,1):spikes_eval(ind,2)),trace_unmix(spikes_eval(ind,1):spikes_eval(ind,2)),'Color',spike_color,'LineWidth',2); 
                    text(frame, double(min(trace_unmix))-1, list_index{ii},'Color',spike_color,'HorizontalAlignment','center','FontSize',14);
                end
            end
%             if mid==num_method
%                 xlabel('Time(frame)');
% %                 xlabel('SNR');
%             end
%             ylabel('SNR');
%             title([list_title{mid,vid},sprintf(', F1=%.4f, corr=%.4f',...
%                 individual_F1{mid-1,vid}(nn),individual_correlation{mid-1,vid}(nn))]);
            title([list_title{mid,vid}],'FontWeight','Normal'); % ,sprintf(', F1=%.4f',individual_F1{mid,vid}(nn))
        end
        hold on;
        trace_max = max(trace_unmix);
        trace_min = min(trace_unmix);
%         marker_pos = trace_max + (trace_max-trace_min) *0.1;
%         plot(spikes_frames,marker_pos,'vm')
    end
end


linkaxes(ax,'x');
linkaxes(ax(2:end),'y');
% set(ax{1},'FontSize',12);
set(ax,'FontSize',12);
xticks(ax,[]);
% yticks(ax,[]);
% ylim(ax,[-3,15]) % *sigma(1)
% xticklabels(ax(1:end-2),[]);
% xticklabels(ax(end-1:end),'auto');
% yticklabels('auto');
% suptitle(['Neuron ',num2str(nno), ' of Video ',Exp_ID(end)]);

pos1=1000;
pos2=5;
plot(pos1+[0,150],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+5*[0,1],'k','LineWidth',2);
text(pos1-90,pos2+2,{'SNR','5'},'HorizontalAlignment','center','FontSize',14); % ,'rotation',90
text(pos1,pos2-2,'5 s','FontSize',14,'rotation',0);

axes(ax(1));
pos1=1000;
pos2=0.2;
plot(pos1+[0,150],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+0.2*[0,1],'k','LineWidth',2);
text(pos1-100,pos2+0.1,{'\Delta{\itF/F}','0.5'},'HorizontalAlignment','center','FontSize',14); %,'rotation',90
text(pos1,pos2-0.06,'5 s','FontSize',14,'rotation',0);

saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' 0809.png']);
saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' 0809.emf']);
