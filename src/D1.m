
function D1()

clc; close all;

tpu_name    = 'oblong_TPU_papillarray_single.mat';
rubber_name = 'oblong_rubber_papillarray_single.mat';

Dcols = 13:15;  % [Dx Dy Dz]

outDir = fullfile(pwd, 'D_outputs');
if ~exist(outDir, 'dir'); mkdir(outDir); end

% LOAD
tpu_path    = findMatFile(tpu_name);
rubber_path = findMatFile(rubber_name);

S_tpu = load(tpu_path);
S_rub = load(rubber_path);

% variables:
X_tpu_full = S_tpu.sensor_matrices_displacement;
X_rub_full = S_rub.sensor_matrices_displacement;

sel_tpu = getSelectedIdx(S_tpu);
sel_rub = getSelectedIdx(S_rub);

% Peak samples 
X_tpu = X_tpu_full(sel_tpu, Dcols);
X_rub = X_rub_full(sel_rub, Dcols);

fprintf('peak samples: TPU=%d, Rubber=%d (total=%d)\n', size(X_tpu,1), size(X_rub,1), size(X_tpu,1)+size(X_rub,1));

%  D.1(b)
fig = figure('Color','w','Position',[80 80 980 640]);
scatter3(X_tpu(:,1), X_tpu(:,2), X_tpu(:,3), 55, 'filled'); hold on;
scatter3(X_rub(:,1), X_rub(:,2), X_rub(:,3), 55, 'filled');
grid on; view(35,18);
xlabel('D\_X (central)');
ylabel('D\_Y (central)');
zlabel('D\_Z (central)');
title('D.1(b): Displacement');
legend({'TPU','Rubber'}, 'Location','southeast');
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir, 'D1b_oblong_disp_3D.png'), 'Resolution', 200);

% Standardise
X_all = [X_tpu; X_rub];
Xz_all = zscoreNoToolbox(X_all);
nT = size(X_tpu,1);
Xz_tpu = Xz_all(1:nT,:);
Xz_rub = Xz_all(nT+1:end,:);

% Labels
y = [ones(nT,1); -ones(size(X_rub,1),1)];

% D.1(c)
pairs = [1 2; 1 3; 2 3];
pairNames = {'D\_X vs D\_Y','D\_X vs D\_Z','D\_Y vs D\_Z'};
axisNames = {'X','Y','Z'};

for k = 1:size(pairs,1)
    idx = pairs(k,:);
    X2 = Xz_all(:, idx);

    [w2, b2] = fisherLDA2class(X2, y);

    fig = figure('Color','w','Position',[90 90 980 640]);
    scatter(Xz_tpu(:,idx(1)), Xz_tpu(:,idx(2)), 55, 'filled'); hold on;
    scatter(Xz_rub(:,idx(1)), Xz_rub(:,idx(2)), 55, 'filled');
    grid on;

    % Set limits
    pad = 0.12;
    xl = padLimits([Xz_tpu(:,idx(1)); Xz_rub(:,idx(1))], pad);
    yl = padLimits([Xz_tpu(:,idx(2)); Xz_rub(:,idx(2))], pad);
    xlim(xl); ylim(yl);

    % Decision boundary
    plotDecisionBoundary2DClipped(gca, w2, b2);

    m = mean(X2,1);
    w2n = w2(:) / max(norm(w2), eps);
    quiver(m(1), m(2), w2n(1), w2n(2), 0.6, 'k', 'LineWidth',2, 'MaxHeadSize',1.2);

    xlabel(['D\_' axisNames{idx(1)} ' (z-scored)']);
    ylabel(['D\_' axisNames{idx(2)} ' (z-scored)']);
    title(['D.1(c): LDA on ' pairNames{k}]);
    legend({'TPU','Rubber','Decision boundary','LD direction'}, 'Location','best');

    removeAxesToolbar(fig);
    exportgraphics(fig, fullfile(outDir, sprintf('D1c_LDA2D_%d%d.png', idx(1), idx(2))), 'Resolution', 200);
end

%  D.1(d)
[w3, b3] = fisherLDA2class(Xz_all, y);

% d-i 
w3n = w3(:) / max(norm(w3), eps);
U = Xz_all * w3n;  % LD axis projection
Vbasis = null(w3n'); % 3x2
v1 = Vbasis(:,1);
Vproj = Xz_all * v1;

U_tpu = U(1:nT); U_rub = U(nT+1:end);
V_tpu = Vproj(1:nT); V_rub = Vproj(nT+1:end);

% Decision line 
u0 = -b3 / max(norm(w3), eps);

fig = figure('Color','w','Position',[100 100 980 640]);
scatter(U_tpu, V_tpu, 55, 'filled'); hold on;
scatter(U_rub, V_rub, 55, 'filled');
grid on;
pad = 0.12;
xlim(padLimits([U_tpu;U_rub], pad));
ylim(padLimits([V_tpu;V_rub], pad));
plot([u0 u0], ylim, 'k--', 'LineWidth',1.8);

xlabel('LD axis (u)');
ylabel('Orthogonal axis (v)');
title('D.1(d.i): 3D LDA reduced to 2D');
legend({'TPU','Rubber','Decision line'}, 'Location','best');

removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir, 'D1di_LDA3D_to2D.png'), 'Resolution', 200);

% d-ii
fig = figure('Color','w','Position',[110 110 980 640]);
scatter3(Xz_tpu(:,1), Xz_tpu(:,2), Xz_tpu(:,3), 55, 'filled'); hold on;
scatter3(Xz_rub(:,1), Xz_rub(:,2), Xz_rub(:,3), 55, 'filled');
grid on; view(35,18);
xlabel('D\_X (z-score)');
ylabel('D\_Y (z-score)');
zlabel('D\_Z (z-score)');
title('D.1(d.ii): LDA discrimination plane in 3D');
plotDecisionPlane3D(w3, b3, Xz_all);
legend({'TPU','Rubber','Decision plane'}, 'Location','southeast');

removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir, 'D1dii_LDA3D_plane.png'), 'Resolution', 200);

end % end main function


% Helper functions

function pathOut = findMatFile(fname)
% Try a few likely locations; otherwise fall back to recursive search.
candidates = {
    fullfile(pwd, fname)
    fullfile(fileparts(pwd), fname)
    fullfile(pwd,'..','..', 'PR_CW_Dataset_2026', 'PR_CW_mat', fname)
    fullfile(pwd,'..','..', 'PR_CW_Dataset_2026', 'PR_CW_mat_files', fname)
    };

for i = 1:numel(candidates)
    if exist(candidates{i}, 'file')
        pathOut = candidates{i};
        return;
    end
end

% Recursive search as last resort
root = pwd;
d = dir(fullfile(root, '**', fname));
if ~isempty(d)
    pathOut = fullfile(d(1).folder, d(1).name);
    return;
end

error('Could not find %s. Put it in the current folder or update findMatFile().', fname);
end

function sel = getSelectedIdx(S)
% Prefer provided 'selected' indices
if isfield(S, 'selected') && ~isempty(S.selected)
    sel = S.selected(:);
    sel = sel(sel>=1);
    return;
end

if ~isfield(S, 'ft_values')
    error('No ''selected'' and no ''ft_values'' found to derive peaks.');
end

fz = S.ft_values(:,3);
sel = simpleLocalMaxima(fz);
% take top 21 peaks if many 
if numel(sel) > 21
    [~,ord] = sort(fz(sel), 'descend');
    sel = sort(sel(ord(1:21)));
end
end

function idx = simpleLocalMaxima(x)
% Toolbox-free peak picking
x = x(:);
n = numel(x);
if n < 3
    idx = 1:n;
    return;
end
idx = find(x(2:n-1) > x(1:n-2) & x(2:n-1) >= x(3:n)) + 1;
if isempty(idx)
    [~,m] = max(x);
    idx = m;
end
end

function [w,b] = fisherLDA2class(X, y)
% Two-class Fisher LDA in standardised feature space.

X1 = X(y==1,:);
X2 = X(y==-1,:);
m1 = mean(X1,1)';
m2 = mean(X2,1)';

S1 = cov(X1, 1);  % normalised by N
S2 = cov(X2, 1);
Sw = S1 + S2;

w = pinv(Sw) * (m1 - m2);
w = w(:);

% Bias 
b = -0.5 * (w'*(m1 + m2));
end

function Xz = zscoreNoToolbox(X)
mu  = mean(X, 1, 'omitnan');
sig = std(X, 0, 1, 'omitnan');
sig(sig==0 | isnan(sig)) = 1;
Xz  = (X - mu) ./ sig;
end

function lim = padLimits(v, padFrac)
v = v(:);
mn = min(v); mx = max(v);
if mn == mx
    mn = mn - 1; mx = mx + 1;
end
r = mx - mn;
lim = [mn - padFrac*r, mx + padFrac*r];
end

function plotDecisionBoundary2DClipped(ax, w, b)
% Draw decision boundary
w = w(:);
xl = xlim(ax); yl = ylim(ax);

a = w(1); bb = w(2); c = b;

pts = [];

% Intersections with vertical edges 
for x0 = [xl(1) xl(2)]
    if abs(bb) > 1e-12
        y0 = -(a*x0 + c)/bb;
        if y0 >= yl(1)-1e-9 && y0 <= yl(2)+1e-9
            pts = [pts; x0 y0]; %#ok<AGROW>
        end
    end
end

% Intersections with horizontal edges 
for y0 = [yl(1) yl(2)]
    if abs(a) > 1e-12
        x0 = -(bb*y0 + c)/a;
        if x0 >= xl(1)-1e-9 && x0 <= xl(2)+1e-9
            pts = [pts; x0 y0]; %#ok<AGROW>
        end
    end
end

% Unique points
if isempty(pts)
    return;
end
pts = unique(round(pts, 12), 'rows');

% Choose two farthest points
if size(pts,1) >= 2
    if size(pts,1) > 2
        best = -inf; p1 = 1; p2 = 2;
        for i=1:size(pts,1)-1
            for j=i+1:size(pts,1)
                d = hypot(pts(i,1)-pts(j,1), pts(i,2)-pts(j,2));
                if d > best
                    best = d; p1 = i; p2 = j;
                end
            end
        end
        pts = pts([p1 p2],:);
    end
    plot(ax, pts(:,1), pts(:,2), 'k--', 'LineWidth', 1.8);
end
end

function plotDecisionPlane3D(w, b, X)
% Plot the plane 
w = w(:);
xl = [min(X(:,1)) max(X(:,1))];
yl = [min(X(:,2)) max(X(:,2))];
zl = [min(X(:,3)) max(X(:,3))];

nGrid = 20;

if abs(w(3)) > 1e-10
    [XX,YY] = meshgrid(linspace(xl(1), xl(2), nGrid), linspace(yl(1), yl(2), nGrid));
    ZZ = -(w(1)*XX + w(2)*YY + b)/w(3);
    h = surf(XX,YY,ZZ,'FaceAlpha',0.18,'EdgeColor','none'); %#ok<NASGU>
elseif abs(w(2)) > 1e-10
    [XX,ZZ] = meshgrid(linspace(xl(1), xl(2), nGrid), linspace(zl(1), zl(2), nGrid));
    YY = -(w(1)*XX + w(3)*ZZ + b)/w(2);
    h = surf(XX,YY,ZZ,'FaceAlpha',0.18,'EdgeColor','none'); %#ok<NASGU>
else
    [YY,ZZ] = meshgrid(linspace(yl(1), yl(2), nGrid), linspace(zl(1), zl(2), nGrid));
    XX = -(w(2)*YY + w(3)*ZZ + b)/w(1);
    h = surf(XX,YY,ZZ,'FaceAlpha',0.18,'EdgeColor','none'); %#ok<NASGU>
end
zlim(zl);
end

function removeAxesToolbar(fig)
% Remove axes toolbar 
try
    axs = findall(fig, 'Type','axes');
    for i = 1:numel(axs)
        try
            axtoolbar(axs(i), {}); %#ok<*AAXES>
        catch
        end
    end
catch
end
end
