%% Part1_B1_PCA.m
% PCA analysis on P4 force data (Cylinder objects only)
% (a) PC display, (b) 2D reduction, (c) 1D number lines

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

shape = 'cylinder';

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

% ---------------- standardise + PCA ----------------
mu    = mean(allData, 1);
sigma = std(allData, 0, 1);
sigma(sigma == 0) = 1;
Xz    = (allData - mu) ./ sigma;

[coeff, score, latent, ~, explained] = pca(Xz);

%% ================= (a) PCA with components =================
figure('Color','w','Name','B1(a) PCA with Principal Components');
hold on; grid on;

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter3(score(idx,1), score(idx,2), score(idx,3), ...
        28, mats{mi}.color, 'filled', 'MarkerFaceAlpha',0.7);
end

quiver3(0,0,0, coeff(1,1), coeff(2,1), coeff(3,1), 2, 'k','LineWidth',2);
quiver3(0,0,0, coeff(1,2), coeff(2,2), coeff(3,2), 2, 'k','LineWidth',2);
quiver3(0,0,0, coeff(1,3), coeff(2,3), coeff(3,3), 2, 'k','LineWidth',2);

xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('B.1(a) PCA of Standardised P4 Force Data (Cylinder)');
legend({'PLA','TPU','Rubber'},'Location','best');
view(3);

savefig(gcf, fullfile(dataDir,'..','assets','B1_1a_cylinder.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','B1_1a_cylinder.png'), 'Resolution',300);

%% ================= (b) 2D PCA replot =================
figure('Color','w','Name','B1(b) 2D PCA - Cylinder');
hold on; grid on;

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter(score(idx,1), score(idx,2), ...
        28, mats{mi}.color, 'filled','MarkerFaceAlpha',0.75);
end

xlabel('PC1'); ylabel('PC2');
title('B.1(b) PCA 2D Projection – Cylinder');
legend({'PLA','TPU','Rubber'},'Location','best');

savefig(gcf, fullfile(dataDir,'..','assets','B1_1b_cylinder.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','B1_1b_cylinder.png'), 'Resolution',300);

%% ================= (c) 1D number line plots =================
figure('Color','w','Name','B1(c) PC Distributions - Cylinder');

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

sgtitle('B.1(c) Distribution Across Principal Components – Cylinder');

savefig(gcf, fullfile(dataDir,'..','assets','B1_1c_cylinder.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','B1_1c_cylinder.png'), 'Resolution',300);

disp('B.1 PCA analysis (Cylinder only) completed.');

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
