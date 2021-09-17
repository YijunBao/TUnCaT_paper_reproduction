function [output,spikes_GT_line] = GT_transient_NAOMi_split(calcium,spikes_frames)
T=size(calcium,2);
hasGT=~cellfun(@isempty, spikes_frames);
num_GT=sum(hasGT);
calcium=calcium(hasGT,:);
[mu_clean, sigma_clean] = SNR_normalization(calcium,'std','median');
peaks_avg = mu_clean;
for nn = 1:num_GT
    trace_clean = calcium(nn,:);
    trace_min = min(trace_clean);
    [peaks,locs] = findpeaks([trace_min,trace_clean,trace_min],'MinPeakProminence',sigma_clean(nn));
    [f,xi] = ksdensity(peaks);
    maxf = max(f); 
    pos = find(f==maxf);
    peaks_avg(nn) = mean(xi(pos)) - mu_clean(nn);
end
thred=mu_clean+peaks_avg/2;
spikes_GT_line=calcium>thred;
output=cell(num_GT,1);
for nn=1:num_GT
    spikes_line=spikes_GT_line(nn,:);
    spikes_line_diff=diff([0,spikes_line,0]);
    starts=find(spikes_line_diff==1);
    ends=find(spikes_line_diff==-1)-1;
    num_active = length(starts);
%     output{nn}=[starts;ends;ones(size(starts))]';
    
    trace=calcium(nn,:);
%     [peaks, locs] = findpeaks(trace, 1:T, 'MinPeakDistance', 5, 'MinPeakProminence', sigma(nn)*3, 'MinPeakHeight', mu(nn)+sigma(nn)*3); %
    [~, locs] = findpeaks(trace, 1:T, 'MinPeakProminence', peaks_avg(nn)/2, 'MinPeakHeight', thred(nn)); %
    locs = [1,locs,T];
    spikes_peaks=false(1,T);
    spikes_peaks(locs)=true;
    list_temp_spikes_split = cell(num_active,1);
    for ii=1:num_active
        num_spikes = sum(spikes_peaks(starts(ii):ends(ii)));
        if num_spikes==0
%             temp_spikes_eval_array(ii,3)=0;
            temp_spikes_split = zeros(0,3);
        elseif num_spikes==1
%             temp_spikes_eval_array(ii,3)=0;
            temp_spikes_split = [starts(ii),ends(ii),0];
        elseif num_spikes>1
%             multi_spike = trace(starts(ii):ends(ii));
            list_locs = locs((locs >= starts(ii)) & (locs <= ends(ii)));
            temp_spikes_split = zeros(num_spikes,3);
            temp_spikes_split(1,1) = starts(ii);
            temp_spikes_split(num_spikes,2) = ends(ii);
            for jj = 1:num_spikes-1
                [~,ind] = min(trace(list_locs(jj):list_locs(jj+1)));
                valley = ind+list_locs(jj)-1;
                temp_spikes_split(jj,2) = valley-1;
                temp_spikes_split(jj+1,1) = valley+1;
            end
        end
        list_temp_spikes_split{ii} = temp_spikes_split;
    end
    output{nn} = cell2mat(list_temp_spikes_split);
%     spikes_eval_array{nn}=temp_spikes_eval_array(logical(temp_spikes_eval_array(:,3)),:);
%     spikes_eval_array{nn}(:,3)=0;
end
end
