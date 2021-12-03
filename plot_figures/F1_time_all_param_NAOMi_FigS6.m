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
list_y = 1.05+(0:5)*0.05;
list_y_log = 60*1.4.^(0:5);
% step = 0.05;
% step_log = 2; % 1.4; % 
step_ns = 0.02;
step_ns_log = 1.2;
yl_time = [1e-1,1e2];
yl_F1 = [0,1.2];
yl_corr = [0,1.2];
list_p = {'n.s.','*','**'};
list_FontSize = [12,14,14];
list_p_cutoff = [0.05, 0.005, 0];
% colors = distinguishable_colors(17);
order = [2,4,5,3];
num_Exp=10;
list_video = {'Raw', 'SNR'};
num_video = length(list_video);
addon = ''; % '_pertmin=0.16_eps=0.1_range'; %  
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
list_legend = {'FISSA','CNMF','Allen SDK','TUnCaT'};

%%
for vid = 1:num_video
    video = list_video{vid};
%     figure('OuterPosition',get(0,'ScreenSize'));
    figure('Position',[00,100,1920,720]);
    gap = [0.04,0.02];  marg_h = 0.10; marg_w = 0.04; 
    ha = tight_subplot(2, num_param_vary, gap, marg_h, marg_w);
    clear ax;

    for ind_param_vary = 1:num_param_vary
        param_vary = list_param_vary{ind_param_vary};
        xtext = list_xtext{ind_param_vary};
        %% 
        load([dir_scores,'\timing_x5_',param_vary,'_GCaMP6f_all_methods_split ',addon,baseline_std,'.mat']);
        if strcmp(param_vary,'sensor')
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
        list_method = list_method(order);
        list_method{4} = 'TUnCaT';
        list_method{3} = 'Allen SDK';

        num_param = length(list_param);
        num_method = length(list_method);
        [recall_CV, precision_CV, F1_CV, time_CV, alpha_CV, thred_ratio_CV, corr_CV] ...
            = deal(zeros(num_Exp,num_method,num_param));
        [Table_recall_CV, Table_precision_CV, Table_F1_CV] = deal(cell(num_method,num_param));

        for mid = 1:length(list_method)
            method = list_method(mid);
        for pid = 1:num_param
            param = list_param(pid);
            list_recall = list_recall_all{mid,vid,pid};
            list_precision = list_precision_all{mid,vid,pid};
            list_F1 = list_F1_all{mid,vid,pid};
            list_corr_unmix = list_corr_unmix_all{mid,vid,pid};
            list_corr_unmix = cellfun(@mean, list_corr_unmix);
            Table_time = Table_time_all{mid,vid,pid};
            list_thred_ratio = list_thred_ratio_all{mid,vid,pid};
            list_recall_2 = reshape(list_recall,num_Exp,[]);
            list_precision_2 = reshape(list_precision,num_Exp,[]);
            list_F1_2 = reshape(list_F1,num_Exp,[]);
            [n1,n2,n3] = size(list_F1);
            if min(n2,n3)>1
                Table_recall_CV{mid,pid} = zeros(n1,n3);
                Table_precision_CV{mid,pid} = zeros(n1,n3);
                Table_F1_CV{mid,pid} = zeros(n1,n3);
                list_alpha = list_alpha_all{mid,vid,pid};
                list_alpha_time = list_alpha_all_time{mid,vid,pid};
            else
                Table_recall_CV{mid,pid} = list_recall;
                Table_precision_CV{mid,pid} = list_precision;
                Table_F1_CV{mid,pid} = list_F1;
            end
            for CV = 1:num_Exp
                train = setdiff(1:num_Exp,CV);
                mean_F1 = squeeze(mean(list_F1_2(train,:),1));
                [~,ind_param] = max(mean_F1);
                recall_CV(CV,mid,pid) = list_recall_2(CV,ind_param);
                precision_CV(CV,mid,pid) = list_precision_2(CV,ind_param);
                F1_CV(CV,mid,pid) = list_F1_2(CV,ind_param);
                if min(n2,n3)>1
                    [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
                    alpha = list_alpha(ind_alpha);
                    alpha_CV(CV,mid,pid) = alpha;
                    thred_ratio_CV(CV,mid,pid) = list_thred_ratio(ind_thred_ratio);
                    ind_alpha = find(list_alpha_time==alpha);
                    time_CV(CV,mid,pid) = Table_time(CV,ind_alpha)+Table_time(CV,end);
                    mean_corr = squeeze(mean(list_corr_unmix(train,:),1));
                    [~,ind_corr] = max(mean_corr);
                    corr_CV(CV,mid,pid) = list_corr_unmix(CV,ind_corr);
                    Table_recall_CV{mid,pid}(CV,:) = permute(list_recall(CV,ind_alpha,:),[1,3,2]);
                    Table_precision_CV{mid,pid}(CV,:) = permute(list_precision(CV,ind_alpha,:),[1,3,2]);
                    Table_F1_CV{mid,pid}(CV,:) = permute(list_F1(CV,ind_alpha,:),[1,3,2]);
                else
                    thred_ratio_CV(CV,mid,pid) = list_thred_ratio(ind_param);
                    if ~isvector(Table_time)
                        Table_time = diag(Table_time);
                    end
                    time_CV(CV,mid,pid) = Table_time(CV);
                    corr_CV(CV,mid,pid) = list_corr_unmix(CV);
                end
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

    %     save(['F1_time_x5_',param_vary,'_GCaMP6f.mat'],'list_param','list_method','list_video',...
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

        axes(ha(sub2ind([num_param_vary, 2], ind_param_vary,1)));
        ax(1,ind_param_vary) = gca;
        hold on;
        h=plot(list_T',F1,'.-','MarkerSize',18,'LineWidth',2);
        h(end).Color = color(5,:);
        for ind = 1:num_method
            errorbar(list_T',F1(:,ind),F1_err(:,ind),F1_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
        end
%         yl2 = [0,0];
%         yl2(1) = floor(min(min(F1-F1_err))/grid_score)*grid_score;
%         yl2(2) = ceil(max(max(F1+F1_err))/grid_score)*grid_score;
%         step = (yl2(2)-yl2(1))*ratio_step;
%         step_ns = step*ratio_ns;
%         list_y = max(max(F1+F1_err))+step*(1:num_method-1);
%         yl2(2) = ceil(list_y(end)/grid_score)*grid_score;
%         ylim(yl2);
    %     yl=get(gca,'Ylim');
    %     list_y_F1 = max(max(F1+F1_err))+step_linear*(1:num_method);
    %     yl2 = [floor(yl(1)/0.1)*0.1,1.2];
    %     ylim(yl2);
        for ind = 1:num_method
            if ind < num_method
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
%         legend(list_method,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
        if ind_param_vary == num_param_vary
            legend(list_legend,'Location','SouthEast','NumColumns',1, 'Interpreter','None') % 'FontSize',12,
        end


        time_all = squeeze(time_CV(:,:,:)); % alpha_CV
        time = squeeze(mean(time_all,1))';
        time_err = squeeze(std(time_all,1,1))';
        axes(ha(sub2ind([num_param_vary, 2], ind_param_vary,2)));
        ax(2,ind_param_vary) = gca;
        hold on;
        h=plot(list_T',time,'.-','MarkerSize',18,'LineWidth',2);
        h(end).Color = color(5,:);
%         set(gca,'FontSize',14);
    %     set(gca,'XScale',xscale);
        set(gca,'YScale','log');
        if ind_param_vary==1
%             ylabel('Processing time (s)');
            ylabel('Speed (neurons\cdotframes/ms)');
        end
%         xticks({});
        xlabel(xtext)
        % xlim()
%         legend(list_method,'Location','NorthOutside','NumColumns',2) % 'FontSize',12,
        for ind = 1:num_method
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
%         list_y = log10(max(max(time+time_err)))+step*(1:num_method-1);
%         yl2(2) = ceil(list_y(end));
%         step_log = 10^(step);
%         step_ns_log = 10^(step_ns);
%         list_y_log = 10.^list_y;
%         yl2 = 10.^yl2;
%         ylim(yl2);
    %     yl=get(gca,'Ylim');
    %     list_y_time = max(max(time+time_err))*step_log.^(1:num_method);
    %     yl2 = [10^floor(log10(yl(1))),10^ceil(log10(list_y_time(end)))];
    %     ylim(yl2);
        for ind = 1:num_method
            time_err_low = time_err(:,ind).*(time_err(:,ind)<=time(:,ind)) + (time(:,ind)-yl_time(1)).*(time_err(:,ind)>time(:,ind));
            errorbar(list_T',time(:,ind),time_err_low,time_err(:,ind),...
                'LineWidth',1,'Color',h(ind).Color,'HandleVisibility','off');
            if ind < num_method
                for jj = 1:num_param
                    p_sign = signrank(time_all(:,ind,jj),time_all(:,end,jj)); % 
                    p_range = find(p_sign > list_p_cutoff,1,'first');
                    if p_range == 1
                        y = list_y_log(ind)*step_ns_log;
                    else
                        y = list_y_log(ind);
                    end
                    text(list_T(jj),y,list_p{p_range},'FontSize',list_FontSize(p_range),...
                        'HorizontalAlignment','Center','Color',h(ind).Color);
                end
            end
        end


        linkaxes(ax(:,ind_param_vary),'x');
        if strcmp(xscale,'log')
            xmin = floor(log10(min(list_T)));
            xmax = ceil(log10(max(list_T)));
            xlim(10.^[xmin,xmax]);
            set(ax(:,ind_param_vary),'XTick',10.^(xmin:xmax));
            set(ax(:,ind_param_vary),'XScale',xscale);
        end
    end
    for row = 1:2
        linkaxes(ax(row,:),'y');
    end
    set(ax,'FontSize',11);
    set(ax(1,:),'XTickLabel',{},'YLim',yl_F1);
    set(ax(2,:),'YLim',yl_time);
    set(ax(:,6),'XTick',1:6);
    set(ax(:,2:num_param_vary),'YTickLabel',{});
    set(ax(:,end),'XTick',list_param);
    set(ax(2,end),'XTickLabel',cellfun(@(x) x(end-1:end), list_prot, 'UniformOutput',false));
    
    set(ax(:,2),'XLim',[5,1000]);
    set(ax(:,5),'XLim',[-20,300]);
    set(ax(:,6),'XLim',[0,6]);
    set(ax,'FontSize',16);
%     for ind_param_vary = 1:num_param_vary
%         set(ax(end,ind_param_vary),'XLabel',list_xtext{ind_param_vary});
%     end
    saveas(gcf,['S6 Fig ',video,'.png']);
%     saveas(gcf,['NAOMi_x4 vary all params ',video,' videos p inverse log 0908.emf']);
end

% legend(list_method,'NumColumns',4,'FontSize',16,'Location','NorthOutside');