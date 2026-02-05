%% ===================== F1: Gaussian Mixture Model (GMM) =====================
% Shape: oblong
% Sensor: P5 (middle papilla)
% Data: Displacement only
% GMM: 3 components

clear; clc; close all;

dataDir = fileparts(mfilename('fullpath'));

% ----- consistent colours for materials -----
COL.PLA    = [0 0.4470 0.7410];
COL.TPU    = [0.8500 0.3250 0.0980];
COL.RUBBER = [0.4660 0.6740 0.1880];

mats = { ...
    struct('name','PLA',   'suffix','',        'color',COL.PLA), ...
    struct('name','TPU',   'suffix','_TPU',    'color',COL.TPU), ...
    struct('name','Rubber','suffix','_rubber', 'color',COL.RUBBER) ...
};

shape = 'oblong';
P5cols = 13:15;   % middle papilla columns (3D)

%% ================= (Data loading & aggregation) =================
allData = [];
labels  = [];

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

    % --- displacement data (Nx27) ---
    dispMat = S.sensor_matrices_displacement;

    % --- use normal force to detect contact peaks ---
    normal = autoPositiveNormal(S.ft_values(:,3));
    N = numel(normal);

    smoothWin = max(3, round(0.003 * N));
    normal_s  = movmean(normal, smoothWin);

    minDist   = max(1, round(0.02 * N));
    minHeight = 0.18 * max(normal_s);

    [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

    % --- extract P5 displacement (central papilla) ---
    P5_disp = dispMat(locs, P5cols);

    allData = [allData; P5_disp];
    labels  = [labels; mi * ones(size(P5_disp,1),1)];
end


%% ================= (a) 2D scatter plot of displacement =================
% Using first two displacement components
figure('Color','w','Name','F1(a) P5 Displacement Scatter');
hold on; grid on;

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter(allData(idx,1), allData(idx,2), 36, mats{mi}.color, 'filled', 'MarkerFaceAlpha',0.7);
end

xlabel('Displacement 1'); ylabel('Displacement 2');
title('F1(a) P5 Displacement Scatter (Oblong)');
legend({'PLA','TPU','Rubber'},'Location','best');

savefig(gcf, fullfile(dataDir,'..','assets','F1a_disp_scatter.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','F1a_disp_scatter.png'),'Resolution',300);

%% ================= (b) Fit GMM & contour plot =================
k = 3; % components
gmm = fitgmdist(allData(:,1:2), k, 'Replicates',10);

% Grid for contour
xrange = linspace(min(allData(:,1)), max(allData(:,1)), 100);
yrange = linspace(min(allData(:,2)), max(allData(:,2)), 100);
[X, Y] = meshgrid(xrange, yrange);
XY = [X(:) Y(:)];

% Evaluate GMM pdf
pdfVals = reshape(pdf(gmm, XY), size(X));

% Plot contour with scatter
figure('Color','w','Name','F1(b) GMM Contour');
hold on; grid on;
contour(X, Y, pdfVals, 10); % 10 contour levels

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter(allData(idx,1), allData(idx,2), 36, mats{mi}.color, 'filled', 'MarkerFaceAlpha',0.7);
end

xlabel('Displacement 1'); ylabel('Displacement 2');
title('F1(b) GMM Contour + P5 Displacement Scatter (Oblong)');
legend({'PLA','TPU','Rubber'},'Location','best');

savefig(gcf, fullfile(dataDir,'..','assets','F1b_gmm_contour.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','F1b_gmm_contour.png'),'Resolution',300);

%% ================= (c) 3D GMM surface plot =================
% Use first two components, height = pdf value
figure('Color','w','Name','F1(c) GMM 3D Surface');
surf(X, Y, pdfVals, 'EdgeColor','none', 'FaceAlpha',0.6);
colormap(jet); hold on; grid on;

for mi = 1:numel(mats)
    idx = labels == mi;
    scatter3(allData(idx,1), allData(idx,2), zeros(sum(idx),1), ...
        36, mats{mi}.color, 'filled', 'MarkerFaceAlpha',0.8);
end

xlabel('Displacement 1'); ylabel('Displacement 2'); zlabel('PDF');
title('F1(c) GMM Surface + P5 Displacement');
view(-45,30); % isometric view

savefig(gcf, fullfile(dataDir,'..','assets','F1c_gmm_surf.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','F1c_gmm_surf.png'),'Resolution',300);

%% ================= (d) Hard cluster assignment =================
[clusterIdx, ~] = cluster(gmm, allData(:,1:2));

figure('Color','w','Name','F1(d) GMM Hard Clusters');
hold on; grid on;

clusterColors = [0.6350 0.0780 0.1840; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330];
markers = {'o','^','s'};

for c = 1:k
    scatter(allData(clusterIdx==c,1), allData(clusterIdx==c,2), ...
        36, clusterColors(c,:), markers{c}, 'filled', 'MarkerFaceAlpha',0.7);
end

xlabel('Displacement 1'); ylabel('Displacement 2');
title('F1(d) Hard Cluster Assignment (Oblong)');
legend({'Cluster 1','Cluster 2','Cluster 3'},'Location','best');

savefig(gcf, fullfile(dataDir,'..','assets','F1d_hard_clusters.fig'));
exportgraphics(gcf, fullfile(dataDir,'..','assets','F1d_hard_clusters.png'),'Resolution',300);

%% ================= (e) Interpretation =================
% Comment in report:
% Compare 'clusterIdx' with 'labels' to see how well clusters match actual materials.
% Usually some overlap occurs; this indicates that displacement alone may not fully separate materials.
% PLA, TPU, and Rubber may overlap in shear displacement space due to similar object response.
disp('F1 GMM analysis completed.');

%% ================= helper functions =================
function normal = autoPositiveNormal(Fz)
    if abs(min(Fz)) > max(Fz)
        normal = -Fz;
    else
        normal = Fz;
    end
end

function [locs, pks] = simplePeaks(x, minDist, minHeight)
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
