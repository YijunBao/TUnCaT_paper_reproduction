addpath('C:\Matlab Files\timer');
addpath('C:\Matlab Files');
warning off;
% pause(3600);
% timer_stop;
timer_start_next;
gcp;
try
%     eval_spikes_ours_SUNS_hasFNFP
%     eval_spikes_CNMF_SUNS_hasFNFP
%     eval_ours_NAOMi_alpha
%     Copy_of_eval_ours_NAOMi_alpha
    eval_spikes_ours;
%     eval_ours_NAOMi
%     eval_1p_ours
%     Copy_of_eval_ours_NAOMi;
%     eval_ours_NAOMi_bin;
%     eval_bgsubs_NAOMi
%     eval_FISSA_NAOMi
%     eval_CNMF_NAOMi
%     eval_ours_NAOMi
%     eval_AllenSDK_NAOMi
%     summary_timing
%     eval_spikes_bgsubs;
%     eval_spikes_FISSA;
%     eval_spikes_CNMF;
%     eval_spikes_ours;
%     eval_spikes_AllenSDK;
%     eval_spikes_ours_bin;
%     prot = 'GCaMP6f'; gen_NAOMi_10;
%     eval_1p_bgsubs
%     eval_1p_FISSA
%     eval_1p_CNMF
%     eval_1p_ours
%     eval_1p_AllenSDK
%     eval_1p_ours_bin;
%     PSNR_simulation_median;
%     PSNR_simulation;
%     eval_spikes_rmoverlap;
%     eval_spikes_prt;
%     eval_spikes_ours_all;
%     eval_MSE_alpha_BinUnmix;
catch
    disp('Failed');
end
% timer_stop;
clear;
%%
% system('shutdown -s'); 
% system('shutdown -a');

%%
% try
%     eval_ours_simulation
%     eval_FISSA_simulation
% catch
%     disp('Failed');
% end

