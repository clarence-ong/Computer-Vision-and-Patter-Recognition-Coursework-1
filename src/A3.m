%% Part1_A3.m  

clear; clc; close all;

dataDir = fileparts(mfilename('fullpath'));

% ----- consistent colours across report -----
COL.PLA    = [0 0.4470 0.7410];   % blue
COL.TPU    = [0.8500 0.3250 0.0980]; % orange
COL.RUBBER = [0.4660 0.6740 0.1880]; % green
% ---------------------------------------------------------------

shapes = {'cylinder','oblong','hexagon'};

% Materials and their filename suffix patterns
mats = { ...
    struct('name','PLA',   'suffix','',        'color',COL.PLA), ...
    struct('name','TPU',   'suffix','_TPU',    'color',COL.TPU), ...
    struct('name','Rubber','suffix','_rubber', 'color',COL.RUBBER) ...
};

% Peak detection params 
smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

P4cols = 13:15;

for s = 1:numel(shapes)
    shape = shapes{s};

    figure('Color','w','Name',['A3 P4 Force Scatter - ' shape]);
    hold on; grid on;

    for mi = 1:numel(mats)
        matinfo = mats{mi};

        % Build filename
        if isempty(matinfo.suffix)
            fname = sprintf('%s_papillarray_single.mat', shape);
        else
            fname = sprintf('%s%s_papillarray_single.mat', shape, matinfo.suffix);
        end
        fpath = fullfile(dataDir,'..', '..', 'PR_CW_Dataset_2026','PR_CW_mat', fname);

        if ~isfile(fpath)
            warning('Missing file: %s (skipped)', fname);
            continue;
        end

        % Load raw trial
        S = load(fpath);

        % Variable names (as per your dataset)
        ft  = S.ft_values;                 % Nx6
        tacF = S.sensor_matrices_force;    % Nx27

        % --- find peak indices  ---
        normal = autoPositiveNormal(ft(:,3));
        N = numel(normal);
        smoothWin = max(3, round(smoothWinFrac * N));
        normal_s = movmean(normal, smoothWin);

        minDist   = max(1, round(minDistFrac * N));
        minHeight = minHeightFrac * max(normal_s);

        [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

        % --- extract P4 force at peak indices ---
        P4_force_at_peaks = tacF(locs, P4cols);  
        Fx = P4_force_at_peaks(:,1);
        Fy = P4_force_at_peaks(:,2);
        Fz = P4_force_at_peaks(:,3);

        % --- plot 3D scatter ---
        scatter3(Fx, Fy, Fz, 28, 'MarkerFaceColor', matinfo.color, ...
            'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.75);

    end

    xlabel('P4 Force X');
    ylabel('P4 Force Y');
    zlabel('P4 Force Z');
    title(sprintf('A.3(a) P4 Force 3D Scatter: %s (PLA vs TPU vs Rubber)', upperFirst(shape)));

    legend({'PLA','TPU','Rubber'}, 'Location','best');
    view(3);

    % Save for report
    savefig(gcf, fullfile(dataDir,'..', '..', 'PR_CW_Dataset_2026','PR_CW_mat', ['A3_P4_scatter_' shape '.fig']));
    exportgraphics(gcf, fullfile(dataDir,'..', '..', 'PR_CW_Dataset_2026','PR_CW_mat', ['A3_P4_scatter_' shape '.png']), 'Resolution', 300);
end

disp('A.3 done. Saved A3_P4_scatter_{cylinder|oblong|hexagon}.png/.fig');

%% ================= helper functions =================
function y = upperFirst(x)
    y = x;
    y(1) = upper(y(1));
end

function normal = autoPositiveNormal(Fz)
% Ensure contact peaks are positive.
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
        keep(end+1,1) = idx; 
        taken = taken | (abs(cand - idx) <= minDist);
    end

    locs = sort(keep);
    pks  = x(locs);
end
