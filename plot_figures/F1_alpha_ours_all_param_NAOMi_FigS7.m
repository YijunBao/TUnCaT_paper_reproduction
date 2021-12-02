clear;
addpath(genpath('..\evaluation'));
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
dir_scores='..\results\NAOMi\evaluation\';
%%
list_param_vary = {'fs', 'T', 'N', 'power', 'Gaus_noise','sensor'}; % 
list_xtext = {'Frame rate (Hz)', 'Video length (s)', 'Number of neurons', ...
    'Laser power (mW)', 'Noise scale', 'GCaMP type'};
num_param_vary = length(list_param_vary);
% grid_score = 0.1;
% grid_time = 5;
% ratio_step = 1/15; 
% ratio_ns = 1/3;
list_y = 1.04+(0:5)*0.04;
% list_y_corr = 160+(0:5)*10;
list_y_time = 70*1.3.^(0:5);
list_y_alpha = 20*1.6.^(0:5);
% step = 0.05;
% step_log = 2; % 1.4; % 
step_ns = 0.02;
% step_ns_corr = 2;
step_ns_time = 1.15;
step_ns_alpha = 1.3;
yl_time = [1e-1,1e2]; % [1e0,1e2]; %
yl_alpha = [1e-2,1e2];
yl_F1 = [0.0,1.2]; % [0.4,1.1]; %
% yl_corr = [0,150];
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14];
list_p_cutoff = [0.05, 0.005, 0];
colors = distinguishable_colors(17);
num_Exp=10;
list_video = {'Raw', 'SNR'}; % 
num_video = length(list_video);
% addon = '_novideounmix_r2_mixout'; % '_pertmin=0.16_eps=0.1_range'; %  
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
order = [1,2]; % 1:5; % 
list_legend = {'Initial \alpha=1','Cross-validation'};
% list_legend = {'\alpha=1','pertmin=0.05','residual=0.01','residual0=0.01','cross-validation'};
% num_addon = length(list_legend);

%%
for vid = 1:num_video
    video = list_video{vid};
    figure('OuterPosition',get(0,'ScreenSize'));
    gap = [0.04,0.02];  marg_h = 0.07; marg_w = 0.04; 
    ha = tight_subplot(3, num_param_vary, gap, marg_h, marg_w);
    clear ax;

    for ind_param_vary = 1:num_param_vary
        param_vary = list_param_vary{ind_param_vary};
        xtext = list_xtext{ind_param_vary};
        %% 
        load([dir_scores,'\timing_',param_vary,'_GCaMP6f_opt_alpha_1 ',baseline_std,'.mat']);
        if strcmp(param_vary,'sensor')
            list_prot = list_param;
            list_param = 1:length(list_prot);
        end

        list_recall_all = list_recall_all(order,:,:);
        list_precision_all = list_precision_all(order,:,:);
        list_F1_all = list_F1_all(order,:,:);
        list_alpha_all = list_alpha_all(order,:,:);
        list_thred_ratio_all = list_thred_ratio_all(order,:,:);
        list_alpha_all_time = list_alpha_all_time(order,:,:);
        list_corr_unmix_all = list_corr_unmix_all(order,:,:);
        Table_time_all = Table_time_all(order,:,:);
        list_addon = list_addon(order);
        final_alpha_all = final_alpha_all(order,:,:);
%         list_addon{5} = 'TUnCaT';
%         list_addon{4} = 'Allen SDK';

        num_param = length(list_param);
        num_addon = length(list_addon);
        [recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV, corr_CV] ...
            = deal(zeros(num_Exp,num_addon,num_param));
        [Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_addon,num_param));

        for aid = 1:length(list_addon)
            addon = list_addon(aid);
        for pid = 1:num_param
            param = list_param(pid);
            list_recall = list_recall_all{aid,vid,pid};
            list_precision = list_precision_all{aid,vid,pid};
            list_F1 = list_F1_all{aid,vid,pid};
            list_corr_unmix = list_corr_unmix_all{aid,vid,pid};
            list_corr_unmix = cellfun(@mean, list_corr_unmix);
            Table_time = Table_time_all{aid,vid,pid};
            list_thred_ratio = list_thred_ratio_all{aid,vid,pid};
            list_recall_2 = reshape(list_recall,num_Exp,[]);
            list_precision_2 = reshape(list_precision,num_Exp,[]);
            list_F1_2 = reshape(list_F1,num_Exp,[]);
            [n1,n2,n3] = size(list_F1);
                Table_recall_CV{aid,pid} = zeros(n1,n3);
                Table_precision_CV{aid,pid} = zeros(n1,n3);
                Table_F1_CV{aid,pid} = zeros(n1,n3);
                list_alpha = list_alpha_all{aid,vid,pid};
                list_alpha_time = list_alpha_all_time{aid,vid,pid};
            for CV = 1:num_Exp
                train = setdiff(1:num_Exp,CV);
                mean_F1 = squeeze(mean(list_F1_2(train,:),1));
                if isvector(mean_F1)
                    mean_F1 = mean_F1';
                end
                [~,ind_param] = max(mean_F1);
                recall_CV(CV,aid,pid) = list_recall_2(CV,ind_param);
                precision_CV(CV,aid,pid) = list_precision_2(CV,ind_param);
                F1_CV(CV,aid,pid) = list_F1_2(CV,ind_param);
                    [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
                    if isscalar(list_alpha)
                        final_alpha = final_alpha_all{aid,vid,pid};
                        if isscalar(final_alpha)
                            alpha = final_alpha;
                            alpha_CV(CV,aid,pid) = alpha;
                        else
                            alpha = final_alpha{CV};
                            alpha_CV(CV,aid,pid) = median(alpha);
                        end
                    else
                        alpha = list_alpha(ind_alpha);
                        alpha_CV(CV,aid,pid) = alpha;
                    end
                    thred_ratio_CV(CV,aid,pid) = list_thred_ratio(ind_thred_ratio);
                    if isscalar(list_alpha_time)
                        ind_alpha = 1;
                    else
                        ind_alpha = find(list_alpha_time==alpha);
                    end
                    time_CV(CV,aid,pid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
                    mean_corr = squeeze(mean(list_corr_unmix(train,:),1));
                    [~,ind_corr] = max(mean_corr);
                    corr_CV(CV,aid,pid) = list_corr_unmix(CV,ind_corr);
                    Table_recall_CV{aid,pid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
                    Table_precision_CV{aid,pid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
                    Table_F1_CV{aid,pid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
            end
        end
        end
        F1_CV(isnan(F1_CV))=0;
        time_CV = 3.*permute(list_N_neuron,[1,3,2])./time_CV; % 
        if strcmp(param_vary,'T')
            time_CV = time_CV.* reshape(list_param-20,[1,1,num_param])/100;
        elseif strcmp(param_vary,'fs')
            time_CV = time_CV.* reshape(list_param,[1,1,num_param])/30;
        end

    %     save(['F1_time_x5_',param_vary,'_GCaMP6f.mat'],'list_param','list_addon','list_video',...
    %         'recall_CV','precision_CV','F1_CV','corr_CV','time_CV','thred_ratio_CV','alpha_CV',...
    %         'Table_recall_CV','Table_precision_CV','Table_F1_CV');

        if strcmp(param_vary,'T')
            list_T = list_param - 20; % 
        elseif contains(param_vary,'noise')
            list_T = list_param*2.7; %  - 20
        else
            list_T = list_param; %  - 20
        end
        if strcmp(param_vary,'T') || strcmp(param_vary,'fs')
            xscale = 'log';
        else
            xscale = 'linear';
        end


    %% F1, time, and correlation
        F1_all = squeeze(F1_CV(:,:,:));
        F1 = squeeze(mean(F1_all,1))';
        F1_err = squeeze(std(F1_all,1,1))';

        axes(ha(sub2ind([num_param_vary, 3], ind_param_vary,1)));
        ax(1,ind_param_vary) = gca;
        hold on;
        h=plot(list_T',F1,'.-','MarkerSize',18,'LineWidth',2);
        h(1).Color = color(4,:);
        h(2).Color = color(5,:);
        for ind = 1:num_addon
            errorbar(list_T',F1(:,ind),F1_err(:,ind),F1_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
        end
%         yl2 = [0,0];
%         yl2(1) = floor(min(min(F1-F1_err))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(F1+F1_err))/grid_score)*grid_score;
%         step = (yl2(2)-yl2(1))*ratio_step;
%         step_ns = step*ratio_ns;
%         list_y = max(max(F1+F1_err))+step*(1:num_addon-1);
%         yl2(2) = ceil(list_y(end)/grid_score)*grid_score;
%         ylim(yl2);
    %     yl=get(gca,'Ylim');
    %     list_y_F1 = max(max(F1+F1_err))+step_linear*(1:num_addon);
    %     yl2 = [floor(yl(1)/0.1)*0.1,1.2];
    %     ylim(yl2);
        for ind = 1:num_addon
            if ind < num_addon
                for jj = 1:num_param
                    p_sign = signrank(F1_all(:,ind,jj),F1_all(:,end,jj)); % 
                    p_range = find(p_sign > list_p_cutoff,1,'first');
                    if p_range == 1
                        y = list_y(ind)+step_ns;
                    else
                        y = list_y(ind);
                    end
                    text(list_T(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
                        'HorizontalAlignment','Center','Color',h(ind).Color);
                end
            end
        end
        if ind_param_vary==1
            ylabel('{\itF}_1');
        end
%         xticks({});
%         xlabel(xtext)
        % xlim()
    %     ylim_current = get(gca,'Ylim');
    %     ylim([ylim_current(1),1.2]);
%         set(gca,'FontSize',14);
    %     set(gca,'XScale',xscale);
%         legend(list_addon,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
        if ind_param_vary == num_param_vary
            legend(list_legend,'Location','SouthEast','NumColumns',1) % 'FontSize',12,, 'Interpreter','None'
        end


        time_all = squeeze(time_CV(:,:,:)); % alpha_CV
        time = squeeze(mean(time_all,1))';
        time_err = squeeze(std(time_all,1,1))';
        axes(ha(sub2ind([num_param_vary, 3], ind_param_vary,3)));
        ax(3,ind_param_vary) = gca;
        hold on;
        h=plot(list_T',time,'.-','MarkerSize',18,'LineWidth',2);
        h(1).Color = color(4,:);
        h(2).Color = color(5,:);
%         set(gca,'FontSize',14);
    %     set(gca,'XScale',xscale);
        set(gca,'YScale','log');
        if ind_param_vary==1
            ylabel('Speed (neurons\cdotframes/ms)');
        end
%         xticks({});
        xlabel(xtext)
        % xlim()
%         legend(list_addon,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
        for ind = 1:num_addon
            errorbar(list_T',time(:,ind),time_err(:,ind),time_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
        end
%         yl2 = [0,0];
%         time_err_low = time-time_err;
%         time_err_low(time_err_low<0)=nan;
%         yl2(1) = floor(log10(min(min(nanmin(nanmin(time_err_low))))));
%         yl2(2) = ceil(log10(max(max(time+time_err))));
%         step = (yl2(2)-yl2(1))*ratio_step;
%         step_ns = step*ratio_ns;
%         list_y = log10(max(max(time+time_err)))+step*(1:num_addon-1);
%         yl2(2) = ceil(list_y(end));
%         step_log = 10^(step);
%         step_ns_log = 10^(step_ns);
%         list_y_log = 10.^list_y;
%         yl2 = 10.^yl2;
%         ylim(yl2);
    %     yl=get(gca,'Ylim');
    %     list_y_time = max(max(time+time_err))*step_log.^(1:num_addon);
    %     yl2 = [10^floor(log10(yl(1))),10^ceil(log10(list_y_time(end)))];
    %     ylim(yl2);
        for ind = 1:num_addon
            time_err_low = time_err(:,ind).*(time_err(:,ind)<=time(:,ind)) + (time(:,ind)-yl_time(1)).*(time_err(:,ind)>time(:,ind));
            errorbar(list_T',time(:,ind),time_err_low,time_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
            if ind < num_addon
                for jj = 1:num_param
                    p_sign = signrank(time_all(:,ind,jj),time_all(:,end,jj)); % 
                    p_range = find(p_sign > list_p_cutoff,1,'first');
                    if p_range == 1
                        y = list_y_time(ind)*step_ns_time;
                    else
                        y = list_y_time(ind);
                    end
                    text(list_T(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
                        'HorizontalAlignment','Center','Color',h(ind).Color);
                end
            end
        end

        
        alpha_all = squeeze(alpha_CV(:,:,:)); % alpha_CV
        alpha = squeeze(mean(alpha_all,1))';
        alpha_err = squeeze(std(alpha_all,1,1))';
        axes(ha(sub2ind([num_param_vary, 3], ind_param_vary,2)));
        ax(2,ind_param_vary) = gca;
        hold on;
        h=plot(list_T',alpha,'.-','MarkerSize',18,'LineWidth',2);
        h(1).Color = color(4,:);
        h(2).Color = color(5,:);
%         set(gca,'FontSize',14);
    %     set(gca,'XScale',xscale);
        set(gca,'YScale','log');
        if ind_param_vary==1
            ylabel('\alpha');
        end
%         xticks({});
%         xlabel(xtext)
        % xlim()
%         legend(list_addon,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
        for ind = 1:num_addon
            errorbar(list_T',alpha(:,ind),alpha_err(:,ind),alpha_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
        end
%         yl2 = [0,0];
%         alpha_err_low = alpha-alpha_err;
%         alpha_err_low(alpha_err_low<0)=nan;
%         yl2(1) = floor(log10(min(min(nanmin(nanmin(alpha_err_low))))));
%         yl2(2) = ceil(log10(max(max(alpha+alpha_err))));
%         step = (yl2(2)-yl2(1))*ratio_step;
%         step_ns = step*ratio_ns;
%         list_y = log10(max(max(alpha+alpha_err)))+step*(1:num_addon-1);
%         yl2(2) = ceil(list_y(end));
%         step_log = 10^(step);
%         step_ns_log = 10^(step_ns);
%         list_y_log = 10.^list_y;
%         yl2 = 10.^yl2;
%         ylim(yl2);
    %     yl=get(gca,'Ylim');
    %     list_y_alpha = max(max(alpha+alpha_err))*step_log.^(1:num_addon);
    %     yl2 = [10^floor(log10(yl(1))),10^ceil(log10(list_y_alpha(end)))];
    %     ylim(yl2);
        for ind = 1:num_addon
            alpha_err_low = alpha_err(:,ind).*(alpha_err(:,ind)<=alpha(:,ind)) + (alpha(:,ind)-yl_alpha(1)).*(alpha_err(:,ind)>alpha(:,ind));
            errorbar(list_T',alpha(:,ind),alpha_err_low,alpha_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
            if ind < num_addon
                for jj = 1:num_param
                    p_sign = signrank(alpha_all(:,ind,jj),alpha_all(:,end,jj)); % 
                    p_range = find(p_sign > list_p_cutoff,1,'first');
                    if p_range == 1
                        y = list_y_alpha(ind)*step_ns_alpha;
                    else
                        y = list_y_alpha(ind);
                    end
                    text(list_T(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
                        'HorizontalAlignment','Center','Color',h(ind).Color);
                end
            end
        end

    %     title_str = [video,' NAOMi videos with different frequency'];
    %     suptitle(title_str);
        linkaxes(ax(:,ind_param_vary),'x');
        if strcmp(xscale,'log')
            xmin = floor(log10(min(list_T)));
            xmax = ceil(log10(max(list_T)));
            xlim(10.^[xmin,xmax]);
            set(ax(:,ind_param_vary),'XTick',10.^(xmin:xmax));
            set(ax(:,ind_param_vary),'XScale',xscale);
        end
    end
    for row = 1:3
        linkaxes(ax(row,:),'y');
    end
    set(ax,'FontSize',11);
    set(ax(1,:),'XTickLabel',{},'YLim',yl_F1);
    set(ax(3,:),'YLim',yl_time);
    set(ax(2,:),'XTickLabel',{},'YLim',yl_alpha);
    set(ax(:,6),'XTick',1:6);
    set(ax(:,2:num_param_vary),'YTickLabel',{});
    set(ax(:,end),'XTick',list_param);
    set(ax(3,end),'XTickLabel',cellfun(@(x) x(end-1:end), list_prot, 'UniformOutput',false));

    set(ax(:,2),'XLim',[5,1000]);
    set(ax(:,5),'XLim',[-20,300]);
    set(ax(:,6),'XLim',[0,6]);
    set(ax,'FontSize',16);

%     for ind_param_vary = 1:num_param_vary
%         set(ax(end,ind_param_vary),'XLabel',list_xtext{ind_param_vary});
%     end
    saveas(gcf,['S7 Fig ',video,'.png']);
    % saveas(gcf,['NAOMi_alpha vary all params ',video,' videos 1 1126.emf']);
end
