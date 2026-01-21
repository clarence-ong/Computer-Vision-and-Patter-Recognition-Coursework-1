%% Part1_B3_PCA.m
% PCA on force data from ALL 9 papillae (27D)
% (a) 2D PCA replot per object shape
% (b) Scree plots for cylinder and oblong

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

shapes = {'cylinder','oblong','hexagon'};

% All papillae force columns (9 papillae × 3 forces)
ALLcols = 1:27;

% Peak detection params (same as previous parts)
smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

for s = 1:numel(shapes)
    shape = shapes{s};

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

        % --- peak detection using normal force ---
        normal = autoPositiveNormal(ft(:,3));
        N = numel(normal);

        smoothWin = max(3, round(smoothWinFrac * N));
        normal_s = movmean(normal, smoothWin);

        minDist   = max(1, round(minDistFrac * N));
        minHeight = minHeightFrac * max(normal_s);

        [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

        % --- extract ALL papillae force vectors ---
        F_all = tacF(locs, ALLcols);

        allData = [allData; F_all];
        labels  = [labels; mi * ones(size(F_all,1),1)];
    end

    % ---------------- standardise + PCA ----------------
    mu    = mean(allData, 1);
    sigma = std(allData, 0, 1);
    sigma(sigma == 0) = 1;
    Xz    = (allData - mu) ./ sigma;

    [coeff, score, latent, ~, explained] = pca(Xz);

    %% ================= (a) 2D PCA replot =================
    figure('Color','w','Name',['B3(a) 2D PCA - ' shape]);
    hold on; grid on;

    for mi = 1:numel(mats)
        idx = labels == mi;
        scatter(score(idx,1), score(idx,2), ...
            28, mats{mi}.color, 'filled','MarkerFaceAlpha',0.75);
    end

    xlabel('PC1'); ylabel('PC2');
    title(['B.3(a) PCA 2D Projection – ' upperFirst(shape)]);
    legend({'PLA','TPU','Rubber'},'Location','best');

    savefig(gcf, fullfile(dataDir,'..','assets',['B3_2D_' shape '.fig']));
    exportgraphics(gcf, fullfile(dataDir,'..','assets',['B3_2D_' shape '.png']), 'Resolution',300);

    %% ================= (b) Scree plot (cylinder & oblong only) =================
    if ismember(shape, {'cylinder','oblong'})
        figure('Color','w','Name',['B3(b) Scree Plot - ' shape]);
        plot(explained, '-o','LineWidth',1.5); grid on;

        xlabel('Principal Component Index');
        ylabel('Variance Explained (%)');
        title(['B.3(b) Scree Plot – ' upperFirst(shape)]);

        savefig(gcf, fullfile(dataDir,'..','assets',['B3_scree_' shape '.fig']));
        exportgraphics(gcf, fullfile(dataDir,'..','assets',['B3_scree_' shape '.png']), 'Resolution',300);
    end
end

disp('B.3 PCA analysis completed.');

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

function y = upperFirst(x)
    y = x;
    y(1) = upper(y(1));
end
