%% Part1_A1.m 
clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
dataDir   = scriptDir;

cylFile = fullfile(dataDir, 'cylinder_papillarray_single.mat');
hexFile = fullfile(dataDir, 'hexagon_papillarray_single.mat');

S_cyl = load(cylFile);
S_hex = load(hexFile);

P_cyl = extractEEPosition(S_cyl);
P_hex = extractEEPosition(S_hex);

figure('Color','w','Name','A.1 End-Effector Trajectory (3D)');
hold on; grid on; axis equal;
plot3(P_cyl(:,1), P_cyl(:,2), P_cyl(:,3), 'LineWidth', 2);
plot3(P_hex(:,1), P_hex(:,2), P_hex(:,3), 'LineWidth', 2);

plot3(P_cyl(1,1),P_cyl(1,2),P_cyl(1,3),'o','MarkerSize',8,'LineWidth',1.5);
plot3(P_cyl(end,1),P_cyl(end,2),P_cyl(end,3),'s','MarkerSize',8,'LineWidth',1.5);
plot3(P_hex(1,1),P_hex(1,2),P_hex(1,3),'o','MarkerSize',8,'LineWidth',1.5);
plot3(P_hex(end,1),P_hex(end,2),P_hex(end,3),'s','MarkerSize',8,'LineWidth',1.5);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('End-Effector 3D Trajectory: Cylinder PLA vs Hexagon PLA');
legend({'Cylinder PLA','Hexagon PLA','Cyl start','Cyl end','Hex start','Hex end'}, 'Location','best');
view(3);

% --- comparsion plots ---
t_cyl = (1:size(P_cyl,1))';
t_hex = (1:size(P_hex,1))';

figure('Color','w','Name','A.1 XYZ Components vs Time Index');
subplot(3,1,1); hold on; grid on;
plot(t_cyl, P_cyl(:,1), 'LineWidth', 1.6);
plot(t_hex, P_hex(:,1), 'LineWidth', 1.6);
ylabel('X (m)'); title('Position Components vs Time Index');
legend({'Cylinder PLA','Hexagon PLA'}, 'Location','best');

subplot(3,1,2); hold on; grid on;
plot(t_cyl, P_cyl(:,2), 'LineWidth', 1.6);
plot(t_hex, P_hex(:,2), 'LineWidth', 1.6);
ylabel('Y (m)');

subplot(3,1,3); hold on; grid on;
plot(t_cyl, P_cyl(:,3), 'LineWidth', 1.6);
plot(t_hex, P_hex(:,3), 'LineWidth', 1.6);
ylabel('Z (m)'); xlabel('Time index');

function P = extractEEPosition(S)
    if ~isfield(S, 'end_effector_poses')
        error(' "end_effector_poses" not exist');
    end
    ee = S.end_effector_poses;

    if isnumeric(ee)
        P = ee(:,1:3);
    elseif iscell(ee)
        eeMat = cell2mat(ee(:));
        P = eeMat(:,1:3);
    elseif isstruct(ee)
        if isfield(ee,'position'), P = ee.position;
        elseif isfield(ee,'pos'),  P = ee.pos;
        else, error('end_effector_poses no position/pos');
        end
    else
        error('not support end_effector_poses: %s', class(ee));
    end
end
