function G1()
clc; close all;

outDir = fullfile(pwd,'G_outputs');
if ~exist(outDir,'dir'); mkdir(outDir); end

%Load datasets
files = dir('*.mat');
if isempty(files)
    error('No .mat files found in current folder.');
end

% Keep only the coursework files
keep = false(numel(files),1);
for i=1:numel(files)
    nm = files(i).name;
    keep(i) = contains(nm,'papillarray_single') && endsWith(nm,'.mat');
end
files = files(keep);
if isempty(files)
    error('No *papillarray_single.mat files found in current folder.');
end

% Build X  and Y 
X = [];
Y = strings(0,1);
classLong = strings(numel(files),1);
classShort = strings(numel(files),1);

for i=1:numel(files)
    S = load(files(i).name);
    if ~isfield(S,'sensor_matrices_displacement')
        error('File %s missing sensor_matrices_displacement', files(i).name);
    end

    % Peak/selected indices
    sel = getSelectedIdx_toolboxFree(S);

    Xi = S.sensor_matrices_displacement(sel,:);

    base = erase(files(i).name,'.mat');
    classLong(i)  = base;                              
    classShort(i) = makeShortName(base);               

    X = [X; Xi]; %#ok<AGROW>
    Y = [Y; repmat(classLong(i), size(Xi,1), 1)]; %#ok<AGROW>
end

fprintf('Loaded samples: %d (per object ~%d)\n', size(X,1), round(size(X,1)/max(numel(files),1)));


[classesLong, ~, yId] = unique(Y,'stable');
K = numel(classesLong);

shortMap = containers.Map(cellstr(classesLong), cellstr(classShort));
classesShort = strings(size(classesLong));
for k=1:K
    classesShort(k) = string(shortMap(char(classesLong(k))));
end

%Train / Test split
% 70/30 split 
holdout = 0.30;
[idxTrain, idxTest] = stratifiedHoldout(yId, holdout, 1);

Xtr_raw = X(idxTrain,:); ytr = yId(idxTrain);
Xte_raw = X(idxTest,:);  yte = yId(idxTest);

%Standardise features
[Xtr_z, muZ, sigZ] = zscoreFit(Xtr_raw);
Xte_z = zscoreApply(Xte_raw, muZ, sigZ);

%PCA via SVD 
[coeff, explained, muP] = pca_svd_fit(Xtr_z);

Ztr_all = (Xtr_z - muP) * coeff;
Zte_all = (Xte_z - muP) * coeff;

cumvar = cumsum(explained);
% target 95%
target = 95;
nPC = find(cumvar >= target, 1, 'first');
if isempty(nPC); nPC = size(Ztr_all,2); end

% Plot cumulative variance
fig = figure('Color','w','Position',[80 80 900 520]);
plot(1:numel(cumvar), cumvar, 'LineWidth', 2);
grid on;
xlabel('Number of PCs');
ylabel('Cumulative explained variance (%)');
title(sprintf('G: PCA cumulative variance (nPC=%d, target=%d%%)', nPC, target));
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir,'G_PCA_cumvar.png'), 'Resolution', 200);

% Use reduced features
Xtr = Ztr_all(:,1:nPC);
Xte = Zte_all(:,1:nPC);

%Bagging

rng(1);

nTrees   = 200;
maxDepth = 8;
minLeaf  = 3;

trees = cell(nTrees,1);
oobVotes = zeros(numel(ytr), K);
oobSeen  = zeros(numel(ytr), 1);
oobErr   = nan(nTrees,1);

for t=1:nTrees
    % bootstrap samples
    N = size(Xtr,1);
    bootIdx = randi(N, N, 1);
    oobMask = true(N,1);
    oobMask(bootIdx) = false;

    trees{t} = trainTree(Xtr(bootIdx,:), ytr(bootIdx), maxDepth, minLeaf);

    % OOB prediction update
    if any(oobMask)
        yhat_oob = predictTree(trees{t}, Xtr(oobMask,:));
        oobVotes(oobMask,:) = oobVotes(oobMask,:) + onehot(yhat_oob, K);
        oobSeen(oobMask) = oobSeen(oobMask) + 1;

        yhat_all = oobMajority(oobVotes, oobSeen);
        valid = oobSeen>0;
        oobErr(t) = mean(yhat_all(valid) ~= ytr(valid));
    else
        % no OOB this round 
        if t>1, oobErr(t) = oobErr(t-1); end
    end
end

% Plot OOB error vs number of trees
fig = figure('Color','w','Position',[90 90 900 520]);
plot(1:nTrees, oobErr, 'LineWidth', 2);
grid on;
xlabel('Number of trees');
ylabel('OOB classification error');
title(sprintf('G(a): OOB error vs trees (nTrees=%d, maxDepth=%d)', nTrees, maxDepth));
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir,'G_a_OOB_error.png'), 'Resolution', 200);

%Predict 
% majority vote across all trees
votesTe = zeros(size(Xte,1), K);
for t=1:nTrees
    yhat = predictTree(trees{t}, Xte);
    votesTe = votesTe + onehot(yhat, K);
end
[~, ypred] = max(votesTe, [], 2);
acc = mean(ypred == yte);

%Confusion matrix 
C = confusionCounts(yte, ypred, K);

fig = figure('Color','w','Position',[120 120 980 760]);
plotConfusion(C, classesLong);
sgtitle(sprintf('G(c): Confusion matrix (Accuracy %.2f%%)', 100*acc));
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir,'G_c_confusion_matrix.png'), 'Resolution', 220);

%Tree visualisation

maxDisplayDepth = 5; % <= complexity

fig = figure('Color','w','Position',[80 80 1400 700]);
plotTreeDigraphClean(trees{1}, classesShort, maxDisplayDepth);
title(sprintf('G(b): Bagged decision tree #1 (display depth ≤ %d)', maxDisplayDepth));
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir,'G_b_tree_1.png'), 'Resolution', 220);

fig = figure('Color','w','Position',[80 80 1400 700]);
plotTreeDigraphClean(trees{2}, classesShort, maxDisplayDepth);
title(sprintf('G(b): Bagged decision tree #2 (display depth ≤ %d)', maxDisplayDepth));
removeAxesToolbar(fig);
exportgraphics(fig, fullfile(outDir,'G_b_tree_2.png'), 'Resolution', 220);

fprintf('Done. Test accuracy = %.2f%%\n', 100*acc);

end

%Helper functions

function short = makeShortName(base)

short = regexprep(base, '_papillarray_single$', '');
end

function sel = getSelectedIdx_toolboxFree(S)

if isfield(S,'selected') && ~isempty(S.selected)
    sel = S.selected(:);
    sel = sel(sel>=1);
    return;
end

if ~isfield(S,'ft_values')
    error('No ''selected'' and no ''ft_values'' found to derive peaks.');
end

fz = autoPositiveNormal(S.ft_values(:,3));
N  = numel(fz);

% Peak detection params 
smoothWinFrac = 0.003;
minDistFrac   = 0.02;
minHeightFrac = 0.18;

smoothWin = max(3, round(smoothWinFrac * N));
fz_s = movmean(fz, smoothWin);

minDist   = max(1, round(minDistFrac * N));
minHeight = minHeightFrac * max(fz_s);

[locs, pks] = simplePeaks(fz_s, minDist, minHeight);

% Fallback
if isempty(locs)
    [~,m] = max(fz_s);
    locs = m;
    pks  = fz_s(m);
end

% Keep at most 21 peaks
nKeep = 21;
if numel(locs) > nKeep
    [~,ord] = sort(pks, 'descend');
    locs = locs(ord(1:nKeep));
end

sel = sort(locs(:));
end

function normal = autoPositiveNormal(Fz)
% Ensure normal force is positive
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

function idx = simpleLocalMaxima(x)

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

function [Xz, mu, sig] = zscoreFit(X)
% Fit z-score parameters on X and return transformed X.
mu  = mean(X, 1, 'omitnan');
sig = std(X, 0, 1, 'omitnan');
sig(sig==0 | isnan(sig)) = 1;
Xz  = (X - mu) ./ sig;
end

function Xz = zscoreApply(X, mu, sig)
% Apply pre-fit z-score parameters.
sig(sig==0 | isnan(sig)) = 1;
Xz = (X - mu) ./ sig;
end

function [coeff, explained, muP] = pca_svd_fit(Xz)

muP = mean(Xz, 1, 'omitnan');
Xc  = Xz - muP;

[~, S, V] = svd(Xc, 'econ');
coeff = V;

% explained variance 
sing = diag(S);
latent = (sing.^2) / max(size(Xc,1)-1, 1);
explained = 100 * latent / sum(latent);
end

function [idxTrain, idxTest] = stratifiedHoldout(yId, holdoutFrac, seed)
% yId: integer class ids 1..K
rng(seed);
N = numel(yId);
idxTrain = false(N,1);
idxTest  = false(N,1);
classes = unique(yId);
for c = classes(:)'
    idx = find(yId==c);
    n = numel(idx);
    nTest = max(1, round(holdoutFrac*n));
    rp = idx(randperm(n));
    idxTest(rp(1:nTest)) = true;
    idxTrain(rp(nTest+1:end)) = true;
end
end

function M = onehot(y, K)
% y integer 1..K
M = zeros(numel(y), K);
for i=1:numel(y)
    M(i, y(i)) = 1;
end
end

function yhat = oobMajority(votes, seen)
% votes: N x K; seen: N x 1 counts of votes
[~, yhat] = max(votes, [], 2);
yhat(seen==0) = 1;
end

function C = confusionCounts(yTrue, yPred, K)
C = zeros(K,K);
for i=1:numel(yTrue)
    C(yTrue(i), yPred(i)) = C(yTrue(i), yPred(i)) + 1;
end
end

function plotConfusion(C, classNames)
% Confusion matrix with correct label printing 
imagesc(C);
axis equal tight;
colormap(parula);
colorbar;

K = size(C,1);
ax = gca;
ax.XTick = 1:K;
ax.YTick = 1:K;
ax.XTickLabel = classNames;
ax.YTickLabel = classNames;
ax.TickLabelInterpreter = 'none';   % show underscores correctly
ax.FontSize = 10;
xtickangle(45);

xlabel('Predicted class');
ylabel('True class');

% numbers
for i=1:K
    for j=1:K
        if C(i,j) ~= 0
            text(j,i, sprintf('%d', C(i,j)), 'HorizontalAlignment','center', 'Color','k', 'FontWeight','bold');
        else
            text(j,i, '0', 'HorizontalAlignment','center', 'Color',[0 0 0], 'FontSize',9);
        end
    end
end
end

%Tree training / prediction

function tree = trainTree(X, y, maxDepth, minLeaf)
% Simple binary decision tree using greedy Gini splits.
K = max(y);
node.X = X;
node.y = y;
node.depth = 1;
tree = grow(node, maxDepth, minLeaf, K);
end

function node = grow(node, maxDepth, minLeaf, K)
% Stop if pure / too small / depth exceeded
counts = accumarray(node.y, 1, [K 1]);
node.classCounts = counts(:)';
node.isLeaf = false;
node.feature = [];
node.thresh  = [];
node.left = [];
node.right = [];

if node.depth >= maxDepth || numel(node.y) <= 2*minLeaf || nnz(counts) == 1
    node.isLeaf = true;
    node.predClass = argmax(counts);
    return;
end

% Find best split
[bestFeat, bestThr, bestGain] = bestGiniSplit(node.X, node.y, K, minLeaf);
if isempty(bestFeat) || bestGain <= 0
    node.isLeaf = true;
    node.predClass = argmax(counts);
    return;
end

node.feature = bestFeat;
node.thresh  = bestThr;
node.predClass = argmax(counts);

xcol = node.X(:,bestFeat);
leftMask  = xcol <= bestThr;
rightMask = ~leftMask;

leftNode.X = node.X(leftMask,:);
leftNode.y = node.y(leftMask);
leftNode.depth = node.depth + 1;

rightNode.X = node.X(rightMask,:);
rightNode.y = node.y(rightMask);
rightNode.depth = node.depth + 1;

node.left  = grow(leftNode,  maxDepth, minLeaf, K);
node.right = grow(rightNode, maxDepth, minLeaf, K);

% free memory
node = rmfield(node, {'X','y','depth'});
end

function [feat, thr, gain] = bestGiniSplit(X, y, K, minLeaf)
% Greedy split by Gini impurity decrease.
feat = [];
thr  = [];
gain = -inf;

N = size(X,1);
parentCounts = accumarray(y, 1, [K 1]);
parentG = giniFromCounts(parentCounts);

D = size(X,2);
for d=1:D
    x = X(:,d);
   
    [xs,ord] = sort(x);
    ys = y(ord);

    % cumulative counts for left
    leftCounts = zeros(K,1);
    rightCounts = parentCounts;

    for i=1:N-1
        c = ys(i);
        leftCounts(c)  = leftCounts(c) + 1;
        rightCounts(c) = rightCounts(c) - 1;

        % only consider threshold between distinct x values
        if xs(i) == xs(i+1)
            continue;
        end

        nL = i;
        nR = N - i;
        if nL < minLeaf || nR < minLeaf
            continue;
        end

        gL = giniFromCounts(leftCounts);
        gR = giniFromCounts(rightCounts);
        childG = (nL/N)*gL + (nR/N)*gR;
        thisGain = parentG - childG;

        if thisGain > gain
            gain = thisGain;
            feat = d;
            thr  = 0.5*(xs(i) + xs(i+1));
        end
    end
end

if isinf(gain)
    gain = -inf;
end
end

function g = giniFromCounts(counts)
N = sum(counts);
if N <= 0
    g = 0;
    return;
end
p = counts / N;
g = 1 - sum(p.^2);
end

function k = argmax(v)
[~,k] = max(v);
end

function yhat = predictTree(tree, X)
% Predict class ids for rows of X
N = size(X,1);
yhat = zeros(N,1);
for i=1:N
    yhat(i) = predictOne(tree, X(i,:));
end
end

function c = predictOne(node, x)
while ~node.isLeaf
    if x(node.feature) <= node.thresh
        node = node.left;
    else
        node = node.right;
    end
end
c = node.predClass;
end

% Cleaner tree plot

function plotTreeDigraphClean(tree, classesShort, maxDepthDisplay)


nodes = struct('label',{},'id',{},'depth',{});
edgesS = [];
edgesT = [];

[nextId, nodes, edgesS, edgesT] = walkClean(tree, 1, 1, maxDepthDisplay, nodes, edgesS, edgesT, classesShort);

G = digraph(edgesS, edgesT);

lbl = strings(1, numel(nodes));
for i=1:numel(nodes)
    lbl(i) = nodes(i).label;
end

p = plot(G, 'Layout','layered', 'Direction','down');
p.NodeLabel = lbl;
p.MarkerSize = 5;
p.ArrowSize = 10;

% Make text readable
p.NodeFontSize = 9;
% Slightly spread nodes
try
    p.LayerSpacing = 40;
    p.NodeSpacing = 18;
catch
end
axis off;
end

function [nextId, nodes, edgesS, edgesT] = walkClean(node, id, depth, maxDepthDisplay, nodes, edgesS, edgesT, classesShort)
% Build digraph lists with depth limit.

nodes(end+1).id = id; %#ok<AGROW>
nodes(end).depth = depth;
nodes(end).label = nodeLabelClean(node, classesShort, depth, maxDepthDisplay);

nextId = id;


if depth >= maxDepthDisplay
    return;
end

if ~node.isLeaf
    leftId = nextId + 1;
    edgesS(end+1) = id; %#ok<AGROW>
    edgesT(end+1) = leftId; %#ok<AGROW>
    [nextId, nodes, edgesS, edgesT] = walkClean(node.left, leftId, depth+1, maxDepthDisplay, nodes, edgesS, edgesT, classesShort);

    rightId = nextId + 1;
    edgesS(end+1) = id; %#ok<AGROW>
    edgesT(end+1) = rightId; %#ok<AGROW>
    [nextId, nodes, edgesS, edgesT] = walkClean(node.right, rightId, depth+1, maxDepthDisplay, nodes, edgesS, edgesT, classesShort);
end
end

function s = nodeLabelClean(node, classesShort, depth, maxDepthDisplay)

n = sum(node.classCounts);
[~,k] = max(node.classCounts);
maj = classesShort(k);

if node.isLeaf
    s = sprintf('%s\nn=%d', maj, n);
else
    if depth >= maxDepthDisplay
        % collapsed
        s = sprintf('%s\n...\nn=%d', maj, n);
    else
        s = sprintf('PC%d \x2264 %.3g\nn=%d', node.feature, node.thresh, n);
    end
end
end

function removeAxesToolbar(fig)
try
    axs = findall(fig, 'Type','axes');
    for i = 1:numel(axs)
        try
            axtoolbar(axs(i), {});
        catch
        end
    end
catch
end
end
