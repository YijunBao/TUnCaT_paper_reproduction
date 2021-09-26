warning off;
gcp;

% Evaluate the accuracy of all methods on all dataset represented by F1
% ours = TUnCaT
eval_1p_bgsubs
eval_1p_FISSA
eval_1p_CNMF
eval_1p_AllenSDK
eval_1p_ours
% eval_1p_ours_bin
eval_ABO_bgsubs
eval_ABO_FISSA
eval_ABO_CNMF
eval_ABO_AllenSDK
eval_ABO_ours
% eval_ABO_ours_bin
eval_ABO_ours_fixed_alpha
eval_NAOMi_bgsubs
eval_NAOMi_FISSA
eval_NAOMi_CNMF
eval_NAOMi_AllenSDK
eval_NAOMi_ours
% eval_NAOMi_ours_bin