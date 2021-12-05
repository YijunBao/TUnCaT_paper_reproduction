clear;
% Set the file containing the evaluated F1 scores
load('..\results\ABO\evaluation\scores_split_ours_RawVideo_UnmixSigma_ksd-psd.mat')

[num_Exp, num_alpha, num_thred_ratio] = size(list_F1);
[recall_CV, precision_CV, F1_CV, alpha_CV, thred_ratio_CV] = deal(zeros(num_Exp,1));

list_recall_2 = reshape(list_recall,num_Exp,[]);
list_precision_2 = reshape(list_precision,num_Exp,[]);
list_F1_2 = reshape(list_F1,num_Exp,[]);
[n1,n2,n3] = size(list_F1);
for CV = 1:num_Exp
    train = setdiff(1:num_Exp,CV);
    mean_F1 = squeeze(mean(list_F1_2(train,:),1));
    [val,ind_param] = max(mean_F1);
    recall_CV(CV) = list_recall_2(CV,ind_param);
    precision_CV(CV) = list_precision_2(CV,ind_param);
    F1_CV(CV) = list_F1_2(CV,ind_param);
    if min(n2,n3)>1
        [ind_alpha,ind_thred_ratio] = ind2sub([n2,n3],ind_param);
        alpha = list_alpha(ind_alpha);
        alpha_CV(CV) = alpha;
        thred_ratio_CV(CV) = list_thred_ratio(ind_thred_ratio);
    else
        thred_ratio_CV(CV) = list_thred_ratio(ind_param);
    end
end

%%
% Table summarizing the cross-validation results, 
% including the optimal parameters (alpha and th_SNR)
% and accuracy metrics (recall, precision, and F1).
array = [(0:num_Exp-1)',alpha_CV,thred_ratio_CV,recall_CV,precision_CV,F1_CV];
array_ext = [array; mean(array,1); std(array,1,1)];
round = [arrayfun(@num2str,(0:num_Exp-1)','UniformOutput',false);'mean';'SD'];
results_table = table(string(round),array_ext(:,2),array_ext(:,3),array_ext(:,4),array_ext(:,5),array_ext(:,6),...
    'VariableNames',{'round','alpha','th_SNR','recall','precision','F1'})
