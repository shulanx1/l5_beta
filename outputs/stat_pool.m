datapath = fullfile(pwd,'PFF_simulation', 'L5');
filename = 'input085_percentage03_no_stochastic';

amp = cell(1,4);
dur = cell(1,4);
% num = zeros(6,4);
for i = 1
control_file = load(fullfile(datapath,[sprintf('apicinh%s_%d_stat', filename, i),'.mat']));
control_raw = load(fullfile(datapath,[sprintf('apicinh%s_%d', filename, i),'.mat']));
amp{1} = control_file.beta_amp;
% amp{1} = abs(control_raw.lfp_beta(control_raw.betaBurst(2,:))-mean(control_raw.lfp_beta));
dur{1} = control_file.beta_dur;
% num(i+1, 1) = control_file.beta_num;
ap5_file = load(fullfile(datapath,[sprintf('noapicalNMDA%s_%d_stat', filename, i),'.mat']));
ap5_raw = load(fullfile(datapath,[sprintf('noapicalNMDA%s_%d', filename, i),'.mat']));
amp{2} = ap5_file.beta_amp;
% amp{2} = abs(ap5_raw.lfp_beta(ap5_raw.betaBurst(2,:))-mean(ap5_raw.lfp_beta));
dur{2} = ap5_file.beta_dur;
% num(i+1, 2) = ap5_file.beta_num;
noCa_file = load(fullfile(datapath,[sprintf('noapicCa%s_%d_stat', filename, i),'.mat']));
noCa_raw = load(fullfile(datapath,[sprintf('noapicCa%s_%d', filename, i),'.mat']));
% amp{3} = abs(noCa_raw.lfp_beta(noCa_raw.betaBurst(2,:))-mean(noCa_raw.lfp_beta));
amp{3} = noCa_file.beta_amp;
dur{3} = noCa_file.beta_dur;
% num(i+1, 3) = noCa_file.beta_num;
lowinh_file = load(fullfile(datapath,[sprintf('control%s_%d_stat', filename, i),'.mat']));
lowinh_raw = load(fullfile(datapath,[sprintf('control%s_%d', filename, i),'.mat']));
amp{4} = lowinh_file.beta_amp;
% amp{4} = abs(lowinh_raw.lfp_beta(lowinh_raw.betaBurst(2,:))-mean(lowinh_raw.lfp_beta));
dur{4} = lowinh_file.beta_dur;
% num(i+1, 4) = lowinh_file.beta_num;
end

for i = 1:4
    amp{i} = amp{i}/1e3;
end

save(fullfile(pwd, 'stat', 'FigureS7F.mat'), 'amp', 'dur')
%%
datapath = fullfile(pwd,'PFF_simulation', 'L23');
filename = 'L23input0';

L23_amp = [];
L23_dur = [];
% num = zeros(6,4);
for i = 1
control_file = load(fullfile(datapath,[sprintf('%s_%d_stat', filename, i),'.mat']));
control_raw = load(fullfile(datapath,[sprintf('%s_%d', filename, i),'.mat']));
L23_amp = abs(control_raw.lfp_beta(control_raw.betaBurst(2,:))-mean(control_raw.lfp_beta))/1e3;
L23_dur = control_file.beta_dur;
end

datapath = fullfile(pwd,'PFF_simulation', 'L5');
filename = 'input085_percentage03_no_stochastic';

L5_amp = [];
L5_dur = [];
% num = zeros(6,4);
for i = 1
control_file = load(fullfile(datapath,[sprintf('apicinh%s_%d_stat', filename, i),'.mat']));
control_raw = load(fullfile(datapath,[sprintf('apicinh%s_%d', filename, i),'.mat']));
L5_amp = abs(control_raw.lfp_beta(control_raw.betaBurst(2,:))-mean(control_raw.lfp_beta));
L5_dur = control_file.beta_dur;
end
save(fullfile(pwd, 'stat', 'FigureS9E.mat'), 'L23_amp', 'L5_amp', 'L23_dur', 'L5_dur')
%%
for n = 1:6
    datapath = fullfile(pwd,sprintf('I_1_E_0_8_apical1_5_basal1_5__%d', n-1)]);
    edges = 0:5:200;
    freq = [1,3,5,10,15,20,25,30,40,60, 0];
    for i = 1:length(freq)
        filename_stat = sprintf('i_mod_%d_stat.mat', freq(i));
        filename = sprintf('i_mod_%d.mat', freq(i));
        load(fullfile(datapath, filename));
        load(fullfile(datapath, filename_stat));
        isi_hist = histcounts(isi, edges);
        % figure, bar(edges(1:end-1) + diff(edges)/2, isi_hist), title(sprintf('%d Hz', freq(i)))
        isi_short(n,i) = sum(isi_hist(find(edges<20)))/sum(isi_hist);
        isi_long(n,i) = sum(isi_hist(find((edges>=60) & (edges < 120))))/sum(isi_hist);
        beta_power(n,i) = sum(abs(hilbert(lfp_beta/1e3)).^2);
        beta_event_count(n,i) = size(betaBurst, 2);
        beta_amplitude(n,i) = mean(beta_amp);
        beta_itpc(n,i) = spike_beta_cohe(40);
        apic_itpc(n,i) = spike_apicbeta_cohe;
    end
end
freq_plot = [1,3,5,10,15,20,25,30,40,60, 100];
figure
yyaxis left
% plot(freq_plot, isi_short)
errorbar(freq_plot,mean(isi_short), std(isi_short)/sqrt(size(isi_short, 1)), 'o', 'MarkerSize',10,  'Color', 'b','MarkerEdgeColor', 'b','MarkerFaceColor', [1,1,1] )
hold on
plot(freq_plot,mean(isi_short), 'b')
yyaxis right
errorbar(freq_plot,mean(isi_long), std(isi_long)/sqrt(size(isi_long, 1)), 'o', 'MarkerSize',10,  'Color', 'r','MarkerEdgeColor', 'r','MarkerFaceColor', [1,1,1] )
hold on
plot(freq_plot, mean(isi_long), 'r')
ax = gca;
ax.YAxis(1).Color = 'b';
ax.YAxis(2).Color = 'r';

figure
yyaxis left
norm_beta_power = [];
for n = 1:size(beta_power, 1)
    norm_beta_power(n,:) = beta_power(n,:)/max(beta_power(n,:));
end
errorbar(freq_plot,mean(norm_beta_power), std(norm_beta_power)/sqrt(size(norm_beta_power, 1)), 'o', 'MarkerSize',10,  'Color', 'b','MarkerEdgeColor', 'b','MarkerFaceColor', [1,1,1] )
hold on
plot(freq_plot,mean(norm_beta_power), 'b')
yyaxis right
norm_beta_event_count = [];
for n = 1:size(beta_event_count, 1)
    norm_beta_event_count(n,:) = beta_event_count(n,:)/max(beta_event_count(n,:));
end
errorbar(freq_plot,mean(beta_event_count/20), std(beta_event_count/20)/sqrt(size(beta_event_count, 1)), 'o', 'MarkerSize',10,  'Color', 'r','MarkerEdgeColor', 'r','MarkerFaceColor', [1,1,1] )
hold on
plot(freq_plot, mean(beta_event_count/20), 'r')
ax = gca;
ax.YAxis(1).Color = 'b';
ax.YAxis(2).Color = 'r';
