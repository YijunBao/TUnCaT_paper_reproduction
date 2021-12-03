%% median std
magenta = [0.9,0.3,0.9];
red = [0.8,0.0,0.0]; 
green = [0.0,0.65,0.0];
list_index = {'(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)'};

std_method = 'quantile-based std comp';
baseline_method = 'median';
% std_method = 'psd';
% baseline_method = 'ksd';

% %%
% list_video = {'SNR','Raw'}; 
% num_video = length(list_video);
% list_method = {'Original trace'; 'Our unmixing'; 'CNMF'};
% num_method = length(list_method);

output_select = find(cellfun(@(x) ~isempty(x), output));
nn=1;
% spike_frames = [212, 697, 2407, 2754, 3458, 3697]; % 26 of 1
% spike_frames = [1559, 3069, 5024]; % 26 of 1
spike_frames = [194, 1968, 3153, 4128, 5025]; % 1 of 9

nno=output_select(nn);
figure('Position',[20,50,900,900]);
[ax, pos] = tight_subplot(num_method,num_video,[0.03,0.04]);
% hold on; 
% clear ax;

% win_filt = length(dFF);
% T = length(raw_traces);
% Tf=T-win_filt+1;
% range=1:T;
% clear mu sigma
% sigma from raw traces
% [mu(1), sigma(1)] = SNR_normalization(list_traces_raw_filt{1}(nno,:),std_method,baseline_method);
% [mu(2), sigma(2)] = SNR_normalization(list_traces_raw_filt{2}(nno,:),std_method,baseline_method);

% name_video = repmat({'SNR video','raw video'},[num_method,1]);
% name_method = repmat(list_method,[1,num_video]);
list_title = list_method;

for vid=1:num_video
    range = 1:size(list_traces_raw{vid},2);
    for mid = 1:num_method
        if mid == 1
            spikes_GT=spikes_GT_array{end,vid}{nn};

            trace_mix=list_traces_raw{vid}(nno,range);
            [mu_mix, sigma_mix] = SNR_normalization(trace_mix,std_method,baseline_method);
            trace_mix=(trace_mix-mu_mix)./sigma_mix; % 

            axes(ax(vid));
            box off
            plot(range,trace_mix); 
            hold on; 
            for ii=1:size(spikes_GT,1)
                transient_range = spikes_GT(ii,1):spikes_GT(ii,2);
                if ~contains(list_video{vid},'SNR')
                    transient_range = transient_range + lag;
                end
                plot(range(transient_range),trace_mix(transient_range),'Color',magenta,'LineWidth',2); 
            end
%             ylabel('SNR');
%             title(list_title{mid,vid});

        else
%             aid = list_ind_alpha(mid-1);
%             alpha = list_alpha(aid);
            spikes_eval=spikes_eval_array{mid,vid}{nn};
            trace_unmix=list_traces_unmixed{mid,vid}(nno,:);
            [mu_unmix, sigma_unmix] = SNR_normalization(trace_unmix,std_method,baseline_method);
%             trace_unmix=(trace_unmix-mu_unmix)./sigma(vid);
            trace_unmix=(trace_unmix-mu_unmix)./sigma_unmix; % 

            axes(ax(num_video*(mid-1)+vid));
            plot(range,trace_unmix); 
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
            end
            
            for ii = 1:length(spike_frames)
                frame = spike_frames(ii);
                ind = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
                if ~isempty(ind)
                    if find(spikes_GT(:,1)<=frame & spikes_GT(:,2)>=frame)
                        spike_color = green;
                    else
                        spike_color = red;
                    end
%                     plot((spikes_eval(ind,1):spikes_eval(ind,2)),trace_unmix(spikes_eval(ind,1):spikes_eval(ind,2)),'Color',spike_color,'LineWidth',2); 
                    text(frame, min(trace_unmix)-3, list_index{ii},'Color',spike_color,'HorizontalAlignment','center','FontSize',14);
                end
            end
%             if mid==num_method
%                 xlabel('Time(frame)');
%             end
%             ylabel('SNR');
        end
        title([list_title{mid,vid}],'FontWeight','Normal'); % ,sprintf(', F1=%.4f',individual_F1{mid,vid}(nn))
    end
end


linkaxes(ax,'y');
ylim(ax,[-10,50]) % *sigma(1)
linkaxes(ax,'x');
% linkaxes(ax{2},'xy');
% set(ax{1},'FontSize',12);
set(ax,'FontSize',12);
% ylim(ax,[-3,15]) % *sigma(1)
% xticklabels(ax(1:end-2),[]);
% xticklabels(ax(1:end-1),[]);
xticks(ax,[]);
% yticks(ax,[]);
% xticklabels(ax(end-1:end),'auto');
% yticklabels('auto');

pos1=500;
pos2=15;
plot(pos1+[0,300],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+20*[0,1],'k','LineWidth',2);
text(pos1-200,pos2+10,{'SNR','20'},'FontSize',14,'HorizontalAlignment','center'); % ,'rotation',90
text(pos1,pos2-5,'10 s','FontSize',14,'rotation',0);

% suptitle(['Neuron ',num2str(nno), ' of Video ',Exp_ID]);
% saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' ksd-psd 0630.png']);
% saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' ksd-psd 0630.emf']);
saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,'.png']);
% saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,'.emf']);
