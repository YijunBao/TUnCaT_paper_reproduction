%% ABO F1 vs alpha for fixed and floating alpha
spike_type = 'ABO'; % {'include','exclude','only'};
num_Exp = 10;
list_addon = {'_novideounmix_r2_mixout1000','_novideounmix_r2_fixed_alpha'};

list_video = {'SNR','Raw'}; % {'SNR','Raw'}; % 
num_video = length(list_video);
num_addon = length(list_addon);
baseline_std = '_ksd-psd'; % '_psd'; % ''; % 
% dir_video='D:\ABO\20 percent 200';
% % dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
% list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
%              '510214538';'524691284';'527048992';'531006860';'539670003'};
[Table_recall_CV, Table_precision_CV, Table_F1_CV, Table_thred_ratio_CV, Table_list_alpha]...
    =deal(cell(num_addon, num_video));

for vid = 1:num_video
    video = list_video{vid};
    for addid = 1:num_addon
        addon = list_addon{addid};
        if strcmp(spike_type, 'NAOMi')
            scorefile = ['scores_split_ours_',simu_opt,'_',video,'Video_Unmix_Sigma',addon,baseline_std,'.mat'];        
        else
            scorefile = ['scores_split_ours_',video,'Video',addon,'_UnmixSigma',baseline_std,'.mat'];
        end
        load(fullfile(spike_type,scorefile));
        num_alpha = length(list_alpha);
        [n1,n2,n3] = size(list_F1);
        [recall_CV, precision_CV, F1_CV, thred_ratio_CV] = deal(zeros(num_Exp,num_alpha));
        for CV = 1:num_Exp
            train = setdiff(1:num_Exp,CV);
            mean_F1 = squeeze(mean(list_F1(train,:,:),1));
            [val,ind_param] = max(mean_F1,[],2);
            for aid = 1:num_alpha
                ind_thred_ratio = ind_param(aid);
                recall_CV(CV,aid) = list_recall(CV,aid,ind_thred_ratio);
                precision_CV(CV,aid) = list_precision(CV,aid,ind_thred_ratio);
                F1_CV(CV,aid) = list_F1(CV,aid,ind_thred_ratio);
                thred_ratio_CV(CV,aid) = list_thred_ratio(ind_thred_ratio);
            end
        end
        Table_recall_CV{addid,vid}=recall_CV;
        Table_precision_CV{addid,vid}=precision_CV;
        Table_F1_CV{addid,vid}=F1_CV;
        Table_thred_ratio_CV{addid,vid}=thred_ratio_CV;
        Table_list_alpha{addid,vid}=list_alpha;
    end
end

save([spike_type,'\F1_split_fix_float_alpha',baseline_std,'.mat'],...
    'list_video','list_addon','Table_thred_ratio_CV','Table_list_alpha',...
    'Table_recall_CV','Table_precision_CV','Table_F1_CV');
