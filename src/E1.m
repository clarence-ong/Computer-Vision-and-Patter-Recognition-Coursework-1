%% ===================== E1: Clustering Analysis =====================
% Shape: oblong
% Sensor: P5 (middle papilla)
% Data: Force only (Fx,Fy,Fz)
% Clustering: K-means with two distance metrics

clear; clc; close all;

dataDir = fileparts(mfilename('fullpath'));

% ----- colours for materials -----
COL.PLA    = [0 0.4470 0.7410];
COL.TPU    = [0.8500 0.3250 0.0980];
COL.RUBBER = [0.4660 0.6740 0.1880];

mats = { ...
    struct('name','PLA',   'suffix','',        'color',COL.PLA), ...
    struct('name','TPU',   'suffix','_TPU',    'color',COL.TPU), ...
    struct('name','Rubber','suffix','_rubber', 'color',COL.RUBBER) ...
};

shape = 'oblong';
P5cols = 13:15;   % middle papilla

smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

allData = [];
labels  = [];

%% ================= (Data loading & aggregation) =================
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

    % Peak detection using normal force
    normal = autoPositiveNormal(ft(:,3));
    N = numel(normal);
    smoothWin = max(3, round(smoothWinFrac*N));
    normal_s = movmean(normal, smoothWin);

    minDist   = max(1, round(minDistFrac*N));
    minHeight = minHeightFrac * max(normal_s);

    [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

    % Extract P5 force vectors
    P5_force = tacF(locs, P5cols);

    allData = [allData; P5_force];
    labels  = [labels; mi * ones(size(P5_force,1),1)];
end

%% ================= (a) Scatter plot of original data =================
figure('Color','w','Name','E1(a) P5 Force Scatter');
hold on; grid on;
for mi = 1:numel(mats)
    idx = labels == mi;
    scatter3(allData(idx,1), allData(idx,2), allData(idx,3), ...
        28, mats{mi}.color, 'filled','MarkerFaceAlpha',0.75);
end
xlabel('Fx'); ylabel('Fy'); zlabel('Fz');
title('E1(a) P5 Force Data Scatter Plot (Oblong)');
legend({'PLA','TPU','Rubber'},'Location','best');
view(3);

savefig(gcf, fullfile(dataDir,'..','assets','E1a_P5_scatter.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','E1a_P5_scatter.png'),'Resolution',300);

%% ================= (b) K-means clustering =================
% Standardised data
Xz = zscore(allData);

k = 3; % number of clusters
rng(1); % reproducibility

[idx, C] = kmeans(Xz, k, 'Replicates',10, 'Distance','sqeuclidean');

% Define cluster colors (new colormap for clarity)
clusterColors = [0.6350 0.0780 0.1840;  % dark red
                 0.4660 0.6740 0.1880;  % green
                 0.3010 0.7450 0.9330]; % cyan
markers = {'o','^','s'}; % different marker shapes

figure('Color','w','Name','E1(b) K-means Clustering');
hold on; grid on;

for c = 1:k
    scatter3(Xz(idx==c,1), Xz(idx==c,2), Xz(idx==c,3), ...
        50, clusterColors(c,:), markers{c}, 'filled','MarkerFaceAlpha',0.7);
end

xlabel('Fx (std)'); ylabel('Fy (std)'); zlabel('Fz (std)');
title('E1(b) K-means Clustering of P5 Force Data (Oblong)');
legend({'Cluster 1','Cluster 2','Cluster 3'},'Location','best');
view(3);

savefig(gcf, fullfile(dataDir,'..','assets','E1b_kmeans_clusters.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','E1b_kmeans_clusters.png'),'Resolution',300);

%% ================= (c) K-means with different distance =================
% Using cosine distance
[idx2, C2] = kmeans(Xz, k, 'Replicates',10, 'Distance','cosine');

figure('Color','w','Name','E1(c) K-means Cosine Distance');
hold on; grid on;

for c = 1:k
    scatter3(Xz(idx2==c,1), Xz(idx2==c,2), Xz(idx2==c,3), ...
        50, clusterColors(c,:), markers{c}, 'filled','MarkerFaceAlpha',0.7);
end

xlabel('Fx (std)'); ylabel('Fy (std)'); zlabel('Fz (std)');
title('E1(c) K-means Clustering (Cosine Distance)');
legend({'Cluster 1','Cluster 2','Cluster 3'},'Location','best');
view(3);

savefig(gcf, fullfile(dataDir,'..','assets','E1c_kmeans_cosine.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','E1c_kmeans_cosine.png'),'Resolution',300);


%% ================= helper functions =================
function normal = autoPositiveNormal(Fz)
% Ensure contact peaks are positive
    Fz = Fz(:);
    if abs(min(Fz)) > max(Fz)
        normal = -Fz;
    else
        normal = Fz;
    end
end

function [locs, pks] = simplePeaks(x, minDist, minHeight)
% Simple peak detector (toolbox-free)
    x = x(:); N = numel(x);
    cand = find(x(2:N-1)>x(1:N-2) & x(2:N-1)>=x(3:N))+1;
    cand = cand(x(cand)>=minHeight);
    [~,o] = sort(x(cand),'descend'); cand=cand(o);
    keep=[]; taken=false(size(cand));
    for i=1:numel(cand)
        if taken(i), continue; end
        keep(end+1)=cand(i);
        taken = taken | abs(cand-cand(i))<=minDist;
    end
    locs=sort(keep); pks=x(locs);
end
