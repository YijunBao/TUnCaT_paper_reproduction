clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));
%%
dir_video='E:\OnePhoton videos\cropped videos\';
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
% list_Exp_ID = list_Exp_ID(1:5);
list_spike_type = {'1p'}; % {'only','include','exclude'};
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'pure','sum'}; % {'pure','sum'}; 

method = 'CNMF'; % {'FISSA','ours','CNMF'}
list_video={'Raw','SNR'};
% video='Raw'; % {'Raw','SNR'}
addon = '_merge';
% list_ndiag={'diag1', 'diag11', 'diag', 'diag02', 'diag22'}; % 
list_ndiag = {'_p1','_p2'}; %, '_diag11'
% list_ndiag = {'+1-3', '+3-0', '+3-1'}; %'diag01', 
% list_ndiag = {'_l1=1.0','_l1=0.0', '_l1=0.2','_l1=0.8'}; %,'_l1=0.8'
list_baseline_std = {'_ksd-psd'}; % '', 
dir_label = [dir_video,'split\'];
num_Exp=length(list_Exp_ID);


for bsid = 1:length(list_baseline_std)
    baseline_std = list_baseline_std{bsid};
% baseline_std = '_ksd-psd'; % ''; % '_psd'; % 
if contains(baseline_std,'psd')
    std_method = 'psd';  % comp
    if contains(baseline_std,'ksd')
        baseline_method = 'ksd';
    else
        baseline_method = 'median';
    end
else
    std_method = 'quantile-based std comp';
    baseline_method = 'median';
end

%%
for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    for ind_ndiag = 1:length(list_ndiag)
        ndiag = list_ndiag{ind_ndiag};
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    list_thred_ratio=100:50:600;% jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                else
                    list_thred_ratio=00:100:800; % jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                end
            else
                list_thred_ratio=4:12; % 6:0.5:9; % 3:3:30; % 
            end
            num_ratio=length(list_thred_ratio);
%         folder = sprintf('traces_ours_%s (tol=1e-4, max_iter=%d)',lower(video),max_iter);
        folder = sprintf('traces_%s_%s%s%s',method,video,ndiag,addon);
%         folder = sprintf('traces_ours');
        dir_FISSA = fullfile(dir_video,folder);
        useTF = strcmp(video, 'Raw');

        [list_recall,list_precision,list_F1]=deal(zeros(num_Exp, num_ratio));

        if useTF
            dFF = h5read('E:\OnePhoton videos\1P_spike_tempolate.h5','/filter_tempolate')';
            dFF = dFF(dFF>exp(-1));
            dFF = dFF'/sum(dFF);
            kernel=fliplr(dFF);
        else
            kernel = 1;
        end

        %%
        fprintf('Neuron Toolbox');
        for ii = 1:num_Exp
            Exp_ID = list_Exp_ID{ii};
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b%12s: ',Exp_ID);
            load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
            if strcmp(spike_type,'exclude')
                for oo = 1:length(output)
                    if ~isempty(output{oo}) && all(output{oo}(:,3))
                        output{oo}=[];
                    end
                end
            elseif strcmp(spike_type,'only')
                for oo = 1:length(output)
                    if ~isempty(output{oo}) && ~all(output{oo}(:,3))
                        output{oo}=[];
                    end
                end
            end

            load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'C_gt','YrA_gt');
            traces_pure = C_gt;
            noise_pure = YrA_gt;
            ncells = size(output,1);
            ncells_remain = size(traces_pure,1);
            if ncells_remain < ncells
                num_miss = ncells - size(traces_pure,1);
                addpath(genpath('C:\Matlab Files\neuron_post'));
                load([dir_video,'\GT Masks merge\FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
                GTMasks_2_permute = reshape(permute(FinalMasks,[2,1,3]),[],ncells);
                load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'A_thr');
                if size(A_thr,2) ~= size(C_gt,1)
                    load(fullfile(dir_FISSA,[Exp_ID,'.mat']),'A_gt');
                    A_thr = A_gt > max(A_gt,[],2)*0.2;
                end
                Dmat = JaccardDist_2(GTMasks_2_permute,A_thr);
                ind_remove = zeros(1,num_miss);
%                     for ind = 1:num_miss
%                         diag_Dmat = diag(Dmat,-ind);
%                         diag_Dmat_smooth = movmean(diag_Dmat,3);
% %                         diff2 = diff(diag_Dmat_smooth,2);
%                         ind_remove(ind) = find(diag_Dmat_smooth<0.85,1)+1;
%                     end           
                di = 0;
                for ix = 1:ncells_remain
                    iy = ix + di;
                    while (iy < ncells) && (Dmat(iy,ix) >= Dmat(iy+1,ix))
                        di = di + 1;
                        ind_remove(di) = ix;
                        iy = iy + 1;
                    end
                end
                while iy < ncells
                    di = di + 1;
                    ind_remove(di) = ix;
                    iy = iy + 1;
                end
%                     if di ~= num_miss
%                         error('Need manual selection');
%                     end
                fp = fopen(sprintf('1p\\removed neurons in %s.txt',Exp_ID),'w');
                fprintf(fp, 'Neuron %d is missing\n',ind_remove); 
                fclose(fp);
                T = size(C_gt,2);
                empty_trace = zeros(1,T);
                empty_noise = median(YrA_gt,1);
                for ind = fliplr(ind_remove)
                    traces_pure = [traces_pure(1:ind-1,:);empty_trace;traces_pure(ind:end,:)];
                    noise_pure = [noise_pure(1:ind-1,:);empty_trace;noise_pure(ind:end,:)];
                end
            end
            traces_sum = traces_pure+noise_pure;
            if useTF
                traces_pure=conv2(traces_pure,kernel,'valid');
                noise_pure=conv2(noise_pure,kernel,'valid');
                traces_sum=conv2(traces_sum,kernel,'valid');
            end
            [~, sigma_pure] = SNR_normalization(noise_pure,std_method,baseline_method);
            [mu_pure, ~] = SNR_normalization(traces_pure,std_method,baseline_method);
            [mu_sum, sigma_sum] = SNR_normalization(traces_sum,std_method,baseline_method);

%                 fprintf('Neuron Toolbox');
            if strcmp(sigma_from,'pure')
                sigma = sigma_pure;
                mu = mu_pure;
                traces_filt = traces_pure;
            else
                sigma = sigma_sum;
                mu = mu_sum;
                traces_filt = traces_sum;
            end

            parfor kk=1:num_ratio
                thred_ratio=list_thred_ratio(kk);
    %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
    %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                [recall, precision, F1,individual_recall,individual_precision,spikes_GT_array,spikes_eval_array]...
                    = GetPerformance_SpikeDetection_split(output,traces_filt,thred_ratio,sigma,mu);
                list_recall(ii,kk)=recall; 
                list_precision(ii,kk)=precision;
                list_F1(ii,kk)=F1;
            end
%             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
        end
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

        %         if num_ratio>1
        %             figure; plot(list_thred_ratio,[list_recall,list_precision,list_F1],'LineWidth',2);
        %             legend('Recall','Precision','F1');
        %             xlabel('thred_ratio','Interpreter','none');
        %             [~,ind]=max(list_F1);
        %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',list_recall(ind),list_precision(ind),list_F1(ind),list_thred_ratio(ind));
        %         else
        %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',recall, precision, F1,thred_ratio);
        %         end
        %%
        save(sprintf('%s\\scores_split_%s_%sVideo%s_%sSigma%s.mat',spike_type,method,video,ndiag,sigma_from,baseline_std),...
            'list_recall','list_precision','list_F1','list_thred_ratio');
        mean_F1 = squeeze(mean(list_F1,1));
        [max_F1, ind_max] = max(mean_F1(:));
        L1 = length(mean_F1);
        disp([list_thred_ratio(ind_max),max_F1])
        fprintf('\b');
        if ind_max == 1
            disp('Decrease thred_ratio');
        elseif ind_max == L1
            disp('Increase thred_ratio');
        end
        end
    end
%     fprintf('\b\b\b\b\b\b\b\b\b\b\b');
    end
end
end