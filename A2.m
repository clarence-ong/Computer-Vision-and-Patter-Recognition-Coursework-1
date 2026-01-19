%% Part1_A2.m  
clear; clc; close all;

dataDir = fileparts(mfilename('fullpath'));

allFiles = dir(fullfile(dataDir, '*_papillarray_single.mat'));
isPLA = arrayfun(@(f) ~contains(f.name,'_TPU_') && ~contains(f.name,'_rubber_'), allFiles);
files = allFiles(isPLA);
assert(~isempty(files), 'No PLA *_papillarray_single.mat found in: %s', dataDir);

smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

allResults = struct();

for k = 1:numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    S = load(fpath);

    ft       = S.ft_values;                    % Nx6
    tactileF = S.sensor_matrices_force;        % Nx27
    tactileD = S.sensor_matrices_displacement; % Nx27

    normal = autoPositiveNormal(ft(:,3));

    % ---- smooth ----
    N = numel(normal);
    smoothWin = max(3, round(smoothWinFrac * N));
    normal_s = movmean(normal, smoothWin);

    % ---- detect peaks ----
    minDist   = max(1, round(minDistFrac * N));
    minHeight = minHeightFrac * max(normal_s);
    [locs, pks] = simplePeaks(normal_s, minDist, minHeight);

    % ---- plot (A.2a) ----
    base = erase(files(k).name, '.mat');
    fig = figure('Color','w','Name',['A2 Peaks - ' base]);
    plot(normal_s,'LineWidth',1.4); grid on; hold on;
    scatter(locs, pks, 45, 'filled');

    xlabel('Time index');
    ylabel('Normal contact force (arb.)');
    title(['A.2(a) Normal-force peaks: ' strrep(base,'_','\_')]);
    legend({'Normal force (smoothed)','Peaks'}, 'Location','best');

    if ~isempty(locs)
        step = max(1, floor(numel(locs)/25));
        for i = 1:step:numel(locs)
            text(locs(i), pks(i), sprintf(' %d', locs(i)), 'FontSize', 8);
        end
    end

    savefig(fig, fullfile(dataDir, ['A2_peaks_' base '.fig']));
    exportgraphics(fig, fullfile(dataDir, ['A2_peaks_' base '.png']), 'Resolution', 300);

    % ---- save peaks & indices (A.2b) ----
    result = struct();
    result.file = files(k).name;
    result.peak_indices = locs(:);
    result.peak_values  = pks(:);
    result.normal_force_smoothed = normal_s(:);

    % ---- extract at peaks (A.2c) ----
    result.ft_at_peaks = ft(locs,:);                 % [numPeaks x 6]
    result.tactile_force_at_peaks = tactileF(locs,:);% [numPeaks x 27]
    result.tactile_disp_at_peaks  = tactileD(locs,:);% [numPeaks x 27]

    safeField = matlab.lang.makeValidName(base);
    allResults.(safeField) = result;

    save(fullfile(dataDir, ['A2_extracted_' base '.mat']), '-struct', 'result');
end

save(fullfile(dataDir, 'A2_all_PLA_peaks_and_extracted.mat'), 'allResults');
disp('A2 done. Saved A2_peaks_*.png/.fig and A2_extracted_*.mat + A2_all_PLA_peaks_and_extracted.mat');

%% ===== helpers =====
function normal = autoPositiveNormal(Fz)
    Fz = Fz(:);
    if abs(min(Fz)) > max(Fz)
        normal = -Fz;
    else
        normal = Fz;
    end
end

function [locs, pks] = simplePeaks(x, minDist, minHeight)
    x = x(:);
    N = numel(x);
    if N < 3, locs = []; pks = []; return; end

    cand = find(x(2:N-1) > x(1:N-2) & x(2:N-1) >= x(3:N)) + 1;
    cand = cand(x(cand) >= minHeight);
    if isempty(cand), locs = []; pks = []; return; end

    [~, order] = sort(x(cand), 'descend');
    cand = cand(order);

    taken = false(size(cand));
    keep = [];

    for i = 1:numel(cand)
        if taken(i), continue; end
        idx = cand(i);
        keep(end+1,1) = idx; %#ok<AGROW>
        taken = taken | (abs(cand - idx) <= minDist);
    end

    locs = sort(keep);
    pks  = x(locs);
end
