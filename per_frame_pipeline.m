%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PER_FRAME_PIPELINE.M
% Trains SVM for per-frame wildfire detection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% --- Paths ---
csvFile = 'Wildfire_Dataset\labels\video_labels.csv';       % CSV with Filename + HasSmoke
videoFolder = 'Wildfire_Dataset\subset_B_videos\all_videos';  

%% --- Parameters ---
framesPerVideo = 10;      % sample frames per video
resizeTo = [224 224];     % target frame size
quantLevels = 16;         % quantization levels for GLCM
rng(42);                  % reproducibility

%% --- Load labels ---
fprintf('Loading labels from %s\n', csvFile);
T = readtable(csvFile);
numVideos = height(T);
fprintf('Found %d videos.\n', numVideos);

%% --- Extract per-frame features ---
featureList = {};
labelList = [];

for i = 1:numVideos
    vidName = T.Filename{i};
    if ismember('HasSmoke', T.Properties.VariableNames)
        lbl = T.HasSmoke(i);
    elseif ismember('HasTrue', T.Properties.VariableNames)
        lbl = T.HasTrue(i);
    else
        lbl = T{i,2};
    end

    % Normalize label to logical
    if iscell(lbl)
        lbl = strcmpi(lbl,'true') | strcmpi(lbl,'1');
    else
        lbl = logical(lbl);
    end

    videoPath = fullfile(videoFolder, vidName);
    if ~isfile(videoPath)
        fprintf('  - Missing %s, skipping\n', vidName);
        continue;
    end

    try
        v = VideoReader(videoPath);
    catch ME
        fprintf('  - Error reading %s: %s\n', vidName, ME.message);
        continue;
    end

    totalFrames = max(1, floor(v.FrameRate*v.Duration));
    idxs = round(linspace(1, totalFrames, min(framesPerVideo, totalFrames)));

    prevFeat = [];
    for fidx = 1:length(idxs)
        frameNumber = idxs(fidx);
        v.CurrentTime = (frameNumber-1)/v.FrameRate;
        frm = readFrame(v);
        frm = imresize(frm, resizeTo);

        % --- Feature extraction ---
        feat = extract_features(frm, quantLevels);  % 15 features

        % --- Delta to previous frame ---
        if isempty(prevFeat)
            deltaFeat = zeros(size(feat));
        else
            deltaFeat = feat - prevFeat;
        end
        prevFeat = feat;

        % --- Combine feat + delta (30 features per frame) ---
        combinedFeat = [feat, deltaFeat];

        % --- Add to dataset ---
        featureList{end+1,1} = combinedFeat;
        labelList(end+1,1) = lbl;
    end
end

%% --- Convert to numeric matrix ---
X = cell2mat(featureList);    % frames x 30
Y = categorical(double(labelList), [1 0], {'Fire','NoFire'});

fprintf('Per-frame dataset: %d frames x %d features\n', size(X,1), size(X,2));

%% --- Train/Test split ---
cv = cvpartition(Y,'HoldOut',0.3);
Xtrain = X(training(cv),:);  Ytrain = Y(training(cv));
Xtest  = X(test(cv),:);      Ytest  = Y(test(cv));

fprintf('Train frames: %d | Test frames: %d\n', size(Xtrain,1), size(Xtest,1));

%% --- Normalize using train statistics ---
mu = mean(Xtrain,1); sigma = std(Xtrain,0,1); sigma(sigma==0)=1;
XtrainZ = (Xtrain - mu) ./ sigma;
XtestZ  = (Xtest  - mu) ./ sigma;

%% --- Train per-frame SVM ---
fprintf('Training per-frame SVM...\n');
SVMModel = fitcsvm(XtrainZ, Ytrain, 'KernelFunction','rbf', 'Standardize', false);

% Fit posterior probabilities for confidence
SVMPosterior = fitPosterior(SVMModel, XtrainZ, Ytrain);

%% --- Evaluate ---
[YPred, score] = predict(SVMPosterior, XtestZ);
conf = score(:, strcmp(SVMPosterior.ClassNames,'Fire'));
acc = mean(YPred == Ytest);
fprintf('Test accuracy: %.2f%%\n', acc*100);

%% --- Save for real-time simulation ---
save('per_frame_SVM.mat', 'SVMPosterior', 'mu', 'sigma', 'quantLevels');

fprintf('Per-frame SVM training completed and saved.\n');
