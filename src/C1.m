%% ===================== C1: t-SNE Analysis =====================
% Apply t-SNE to middle papilla (P5) force data
% Shapes: Cylinder (mandatory), Oblong (chosen)

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

shapes = {'cylinder','oblong'};   % C.1(a): cylinder, C.1(b): oblong
perplexities = [10 40];

% ----- middle papilla (P5) force columns -----
P5cols = 13:15;

for s = 1:numel(shapes)
    shape = shapes{s};

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

        % Ensure contact force is positive
        normal = autoPositiveNormal(ft(:,3));
        normal_s = movmean(normal, max(3, round(0.003*numel(normal))));

        % Peak detection
        minDist   = round(0.02 * numel(normal));
        minHeight = 0.18 * max(normal_s);
        [locs, ~] = simplePeaks(normal_s, minDist, minHeight);

        % Extract P5 force vectors
        P5_force = tacF(locs, P5cols);

        allData = [allData; P5_force];
        labels  = [labels; mi * ones(size(P5_force,1),1)];
    end

    %% ================= (Standardisation) =================
    Xz = zscore(allData);

    %% ================= C.1 t-SNE embedding =================
    for p = 1:numel(perplexities)
        perp = perplexities(p);

        %% ---------- (a)/(b) t-SNE with selected perplexity ----------
        [Y, loss] = tsne(Xz, ...
            'NumDimensions', 2, ...
            'Perplexity', perp, ...
            'Standardize', false);

        %% ================= (a)/(b) 2D t-SNE replot =================
        figure('Color','w', ...
               'Name',sprintf('C1 t-SNE %s (Perplexity %d)',shape,perp));
        hold on; grid on;

        for mi = 1:numel(mats)
            idx = labels == mi;
            scatter(Y(idx,1), Y(idx,2), ...
                30, mats{mi}.color, 'filled', 'MarkerFaceAlpha',0.75);
        end

        title(sprintf('C.1 t-SNE – %s | Perplexity = %d | Loss = %.2f', ...
            upperFirst(shape), perp, loss));
        xlabel('t-SNE 1'); ylabel('t-SNE 2');
        legend({'PLA','TPU','Rubber'},'Location','best');

        % Save figures
        savefig(gcf, fullfile(dataDir,'..','assets', ...
            sprintf('C1_tsne_%s_p%d.fig',shape,perp)));
        exportgraphics(gcf, fullfile(dataDir,'..','assets', ...
            sprintf('C1_tsne_%s_p%d.png',shape,perp)), 'Resolution',300);
    end
end

disp('C1 t-SNE analysis completed.');

%% ================= helper functions =================
function y = upperFirst(x)
    y = x; y(1) = upper(y(1));
end

function normal = autoPositiveNormal(Fz)
% Ensure contact force is positive
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
