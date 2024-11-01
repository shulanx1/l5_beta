%% Figure S7F
load(fullfile(pwd, 'stat','FigureS7F.mat'))
addpath(genpath(fullfile(pwd, 'plotting')))

boxplot_compact(amp)
xticks(1:4), xticklabels({'control','apic NMDA block', 'apic VGCC block', 'less apic inh.'}), ylabel('Beta Event AMP'), ylim([0,3])
%% Figure S9E
load(fullfile(pwd, 'stat','FigureS9E.mat'))
addpath(genpath(fullfile(pwd, 'plotting')))

boxplot_with_datapoint({L23_amp,L5_amp})
xticks([1,2]), xticklabels({'L23','L5'}), ylabel('Beta Event AMP. [uV]')