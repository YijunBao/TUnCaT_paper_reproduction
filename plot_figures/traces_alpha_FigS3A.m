color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
magenta = [0.9,0.3,0.9];
red = [0.8,0.0,0.0]; 
green = [0.0,0.65,0.0];
colors = color([1,3,6],:);
list_index = {'(i)','(ii)','(iii)','(iv)','(v)','(vi)','(vii)'};

%%
dir_traces='..\results\ABO\unmixed traces\';
list_neighbor = [2,1];
median_align = [50,200,350];

% nn=34; % 2 ;
% list_ind_alpha = [0,6, 8,10, 12,14];
% list_ind_alpha = [0,6; 8,10; 12,14];
% list_ind_alpha = [0,6]; % part1
% list_ind_alpha = 10:2:14; % part2
nn=22; % 3;
list_ind_alpha = [0,4, 7,10, 13,16];
% list_ind_alpha = [0,4; 7,10; 13,16];
% list_ind_alpha = [0,4]; % part1
% list_ind_alpha = 10:3:16; % part2

% list_video = {'SNR','Raw'}; 
% num_video = length(list_video);
% list_method = {'Original trace'; 'Our unmixing'; 'CNMF'};
% num_method = length(list_method);
std_method = 'quantile-based std comp';
baseline_method = 'median';
% std_method = 'psd';  % comp
% baseline_method = 'ksd';

output_select = find(cellfun(@(x) ~isempty(x), output));
% nn=17;
% spike_frames = [1251, 3577, 12293, 14776, 17678, 19384, 20357]; 
% spike_frames = [5444, 8670, 16094, 16337, 19851, 22281, 22375]; % 55 of 4 more accurate peaks
% spike_frames = [5445, 8671, 16095, 16339, 19852, 22284, 22375]; % 55 of 4 better display
spike_frames = [];
nno=output_select(nn);
dir_demixtest = [dir_traces,'\demixtest_',num2str(nno),'_',Exp_ID];

[num_figure, num_alpha] = size(list_ind_alpha);
% num_alpha = length(list_ind_alpha);
num_plot = num_alpha;

for fid = num_figure:-1:1
%     figure('Position',[20,950-300*(fid),900,300]);
%     [ax, pos] = tight_subplot(num_plot,num_video,[0.08,0.04]);
    figure('Position',[1020,00,900,1000]);
    [ax, pos] = tight_subplot(num_plot,num_video,[0.04,0.03]);

    % hold on; 
    % clear ax;

    % name_video = repmat({'SNR video','raw video'},[num_method,1]);
    % name_method = repmat(list_method,[1,num_video]);
%     list_title = arrayfun(@(x) ['\alpha = ',num2str(x)], list_alpha(list_ind_alpha(fid,:)), 'UniformOutput',false);
%     list_title = [{'BG subtraction'}; list_title];
    list_title = cell(num_plot,1);
    for mid = 1:num_plot
        aid = list_ind_alpha(fid,mid);
        if aid == 0
            list_title{mid} = 'BG subtraction';
        else
            list_title{mid} = ['\alpha = ',num2str(list_alpha(list_ind_alpha(fid,mid)))];
        end
    end
    list_title = repmat(list_title,[1,num_video]);

    for vid=1:num_video
        range = 1:size(list_traces_raw{vid},2);
        for mid = 1:num_plot
            aid = list_ind_alpha(fid,mid);
            axes(ax(num_video*(mid-1)+vid));
            box off
            hold on; 
            if aid == 0
                spikes_GT=spikes_GT_array{end,vid}{nn};
    %             trace_mix=list_traces_raw{vid}(nno,range);
                load([dir_demixtest,'\demixtest_',num2str(list_alpha(1)),'.mat'],'list_tracein','list_subtrace','demix')

                for neighbor = list_neighbor
                    trace_mix=list_tracein{nno}(neighbor,range);
                    [mu_mix, sigma_mix] = SNR_normalization(trace_mix,std_method,baseline_method);
                    trace_mix = trace_mix-mu_mix+median_align(neighbor);
        %             trace_mix=(trace_mix-mu_mix)./sigma_mix;
                    plot(range,trace_mix,'Color',colors(neighbor,:)); 
                    if neighbor == 1
                        for ii=1:size(spikes_GT,1)
                            transient_range = spikes_GT(ii,1):spikes_GT(ii,2);
                            if ~contains(list_video{vid},'SNR')
                                transient_range = transient_range + lag;
                            end
                            plot(range(transient_range),trace_mix(transient_range),'Color',magenta,'LineWidth',2); 
                        end
                    end
        %             ylabel('SNR');
                end
                title(list_title{mid,vid},'FontWeight','Normal');

            else
                load([dir_demixtest,'\demixtest_',num2str(list_alpha(aid)),'.mat'],'list_tracein','list_subtraces','list_traceout')
                spikes_eval=spikes_eval_array{aid,vid}{nn};
    %             trace_unmix=list_traces_unmixed{aid,vid}(nno,:);

                for neighbor = list_neighbor
                    if ~isempty(list_subtraces) && any(neighbor-1 == list_subtraces{nno})
                        trace_unmix=zeros(1,length(range));
                    else
                        trace_unmix=list_traceout{nno}(neighbor,range);
                    end
                    [mu_unmix, sigma_unmix] = SNR_normalization(trace_unmix,std_method,baseline_method);
                    trace_unmix = trace_unmix-mu_unmix+median_align(neighbor);
        %             trace_unmix=(trace_unmix-mu_unmix)./sigma_unmix;
                    plot(range,trace_unmix,'Color',colors(neighbor,:)); 
                    if (neighbor == 1) && (max(trace_unmix) > min(trace_unmix))
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
                    end

    %                 for ii = 1:length(spike_frames)
    %                     frame = spike_frames(ii);
    %                     ind = find(spikes_eval(:,1)<=frame & spikes_eval(:,2)>=frame); 
    %                     if ~isempty(ind)
    %                         if find(spikes_GT(:,1)<=frame & spikes_GT(:,2)>=frame)
    %                             spike_color = green;
    %                         else
    %                             spike_color = red;
    %                         end
    %     %                     plot((spikes_eval(ind,1):spikes_eval(ind,2)),trace_unmix(spikes_eval(ind,1):spikes_eval(ind,2)),'Color',spike_color,'LineWidth',2); 
    %                         text(frame, min(trace_unmix)-3, list_index{ii},'Color',spike_color,'HorizontalAlignment','center','FontSize',14);
    %                     end
    %                 end
    %             if mid==num_method
    %                 xlabel('Time(frame)');
    %             end
    %             ylabel('SNR');
                end
                title([list_title{mid,vid}],'FontWeight','Normal'); % ,sprintf(', F1=%.4f',individual_F1{mid,vid}(nn))
            end
        end
    end


    linkaxes(ax,'y');
    % ylim(ax,[-10,40]) % *sigma(1)
    linkaxes(ax,'x');
    % linkaxes(ax{2},'xy');
    % set(ax{1},'FontSize',12);
    set(ax,'FontSize',12);
    % ylim(ax,[-3,15]) % *sigma(1)
    % ylim(ax,[-3,15]) % *sigma(1)
    % xticklabels(ax(1:end-2),[]);
    % xticklabels(ax(1:end-1),[]);
    xticks(ax,[]);
    % yticks(ax,[]);
    % xticklabels(ax(end-1:end),'auto');
    % yticklabels('auto');

    pos1=2500;
    pos2=300;
    plot(pos1+[0,1500],pos2*[1,1],'k','LineWidth',2);
    plot(pos1*[1,1],pos2+200*[0,1],'k','LineWidth',2);
    text(pos1-760,pos2+100,{'\Delta{\itF}','200'},'HorizontalAlignment','center','FontSize',14); % ,'rotation',90
    text(pos1,pos2-60,'50 s','FontSize',14,'rotation',0);
    set(ax,'ylim',[000,500])

    % pos1=6000;
    % pos2=100;
    % plot(pos1+[0,1500],pos2*[1,1],'k','LineWidth',2);
    % plot(pos1*[1,1],pos2+100*[0,1],'k','LineWidth',2);
    % text(pos1-760,pos2+50,{'\Delta{\itF}','100'},'HorizontalAlignment','center','FontSize',14); % ,'rotation',90
    % text(pos1,pos2-40,'50 s','FontSize',14,'rotation',0);
    % ylim([-300,200])

    saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' alpha S3 Fig A.png']);
    % saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' alpha neighbor ',mat2str(list_neighbor),' 0908.emf']);
%     saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' alpha part',num2str(fid),' neighbor ',mat2str(list_neighbor),' 0908.png']);
%     saveas(gcf,['Neuron ',num2str(nno), ' of Video ',Exp_ID,' alpha part',num2str(fid),' neighbor ',mat2str(list_neighbor),' 0908.emf']);
end
