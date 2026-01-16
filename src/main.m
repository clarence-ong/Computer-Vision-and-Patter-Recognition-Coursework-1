% Get the directory of the currently running script
currentFilePath = mfilename('fullpath');
[currentDir, ~, ~] = fileparts(currentFilePath);

% Construct path to the data folder (one level above repo)
dataFolder = fullfile(currentDir, '..', '..', 'PR_CW_Dataset_2026','PR_CW_mat');

% Specify .mat file name
matFileName = 'cylinder_papillarray_single.mat';

% Full path to the .mat file
matFilePath = fullfile(dataFolder, matFileName);

% Check that the file exists
if ~isfile(matFilePath)
    error('Data file not found: %s', matFilePath);
end

% Load the .mat file
data = load(matFilePath);

% Display loaded variable names
disp('Loaded variables:');
disp(fieldnames(data));