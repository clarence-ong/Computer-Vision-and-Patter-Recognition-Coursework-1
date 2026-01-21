%% Part1_B2_PCA.m
% PCA analysis on P4 force data (Oblong objects only)
% (b) 2D reduction, (c) 1D number lines

clear; clc; close all;

dataDir = fileparts(mfilename('fullpath'));

% ----- consistent colours across report -----
COL.PLA    = [0 0.4470 0.7410];
COL.TPU    = [0.8500 0.3250 0.0980];
COL.RUBBER = [0.4660 0.6740 0.1880];

mats = { ...
    struct('name','PLA',   'suffix','',        'color',COL.PLA), ...
    struct('name','TPU',   'suffix','_TPU',    'color',COL.TPU), ...
    struct('name','Rubber','suffix','_rubber', 'color',COL.RUBBER) ...
};

shape = 'oblong';

% P4 force columns
P4cols = 13:15;

% Peak detection params (same as Part A.3)
smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

allData = [];
labels  = [];

% ---------------- load & aggregate data ----------------
for mi = 1:numel(mats)
    matinfo = mats{mi};

    if isempty(matinfo.suffix)
        fname = sprintf('%s_papillarray_single.mat', shape);
    else
        fname = sprintf('%s%s_papillarray_single.mat', shape, matinfo.suffix);
    end

    fpath = fullfile(dataDir,'..','..','PR_CW_Dataset_2026','PR_CW_mat',fname);
    if ~isfile(fpath), continue; end

    S = load(fpath);
    ft   = S.ft_values;
    tacF = S.sensor_matrices_force;

    normal = autoPositiveNormal(ft(:,3));
    N = numel(normal);

    smoothWin = max(3, round(smoothWinFrac * N));
    normal_s = movmean(normal, smoothWin);

    minDist   = max(1, round(minDistFrac * N));
    minHeight = minHeightFrac * max(normal_s);

    [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

    P4_force = tacF(locs, P4cols);

    allData = [allData; P4_force];
    labels  = [labels; mi * ones(size(P4_force,1),1)];
end

% ---------------- standardise + PCA (Stats & ML Toolbox) ----------------
mu    = mean(allData, 1);
sigma = std(allData, 0, 1);
sigma(sigma==0) = 1;  % avoid divide-by-zero
Xz    = (allData - mu) ./ sigma;

[coeff, score, latent, ~, explained] = pca(Xz);

%% ================= (b) 2D PCA replot =================
figure('Color','w','Name','B2(b) 2D PCA - Oblong');
hold on; grid on;

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter(score(idx,1), score(idx,2), ...
        28, mats{mi}.color, 'filled','MarkerFaceAlpha',0.75);
end

xlabel('PC1'); ylabel('PC2');
title('B.2(b) PCA 2D Projection – Oblong');
legend({'PLA','TPU','Rubber'},'Location','best');

savefig(gcf, fullfile(dataDir,'..','assets','B2_1b_oblong.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','B2_1b_oblong.png'), 'Resolution',300);

%% ================= (c) 1D number line plots =================
figure('Color','w','Name','B2(c) PC Distributions - Oblong');

numPC = size(score,2);
for pc = 1:numPC
    subplot(numPC,1,pc); hold on; grid on;
    for mi = 1:numel(mats)
        idx = labels == mi;
        scatter(score(idx,pc), zeros(sum(idx),1), ...
            12, mats{mi}.color, 'filled','MarkerFaceAlpha',0.6);
    end
    ylabel(sprintf('PC%d',pc));
    if pc == numPC, xlabel('Component Value'); end
end

sgtitle('B.2(c) Distribution Across Principal Components – Oblong');

savefig(gcf, fullfile(dataDir,'..','assets','B2_1c_oblong.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','B2_1c_oblong.png'), 'Resolution',300);

disp('B.2 PCA analysis (Oblong) completed.');

%% ================= helper functions =================
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
