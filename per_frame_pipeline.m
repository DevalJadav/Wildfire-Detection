%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PER_FRAME_PIPELINE.M
% Trains a per-frame SVM and saves per_frame_SVM.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% CONFIG
csvFile = 'Wildfire_Dataset/labels/video_labels.csv';
videoFolder = 'Wildfire_Dataset/subset_B_videos/all_videos';
framesPerVideo = 10;   % sample per video
resizeTo = [224 224];
quantLevels = 16;
rng(42);

% Read labels table
T = readtable(csvFile);
numVideos = height(T);

features = {};
labels = [];

fprintf('Extracting per-frame features...\n');
for vi = 1:numVideos
    vidName = T.Filename{vi};
    if ismember('HasSmoke', T.Properties.VariableNames)
        lbl = T.HasSmoke(vi);
    else
        lbl = T{vi,2};
    end
    % normalize label to logical
    if iscell(lbl)
        lbl = strcmpi(lbl,'true') | strcmpi(lbl,'1');
    else
        lbl = logical(lbl);
    end

    videoPath = fullfile(videoFolder, vidName);
    if ~isfile(videoPath)
        fprintf('  Missing %s — skipping\n', vidName);
        continue;
    end

    try
        v = VideoReader(videoPath);
    catch ME
        fprintf('  Error reading %s: %s — skipping\n', vidName, ME.message);
        continue;
    end

    totalFrames = max(1, floor(v.FrameRate * v.Duration));
    idxs = round(linspace(1, totalFrames, min(framesPerVideo, totalFrames)));
    prevFeat = [];

    for k = 1:length(idxs)
        fnum = idxs(k);
        v.CurrentTime = max((fnum-1)/v.FrameRate, 0);
        frame = readFrame(v);
        frame = imresize(frame, resizeTo);

        feat = extract_features(frame, quantLevels);   % 15-dim
        if isempty(prevFeat)
            delta = zeros(size(feat));
        else
            % safeguard same length
            m = min(numel(feat), numel(prevFeat));
            delta = feat(1:m) - prevFeat(1:m);
            if numel(delta) < numel(feat)
                delta = [delta, zeros(1, numel(feat)-numel(delta))];
            end
        end
        prevFeat = feat;

        combined = [feat, delta];   % 30 dims (15 + 15)
        features{end+1,1} = combined;
        labels(end+1,1) = lbl;
    end
end

% Convert
X = cell2mat(features); % N x 30
Y = categorical(double(labels), [1 0], {'Fire','NoFire'});

fprintf('Dataset ready: %d frames x %d features\n', size(X,1), size(X,2));

% Train/test split
cv = cvpartition(Y,'HoldOut',0.25);
Xtrain = X(training(cv),:); Ytrain = Y(training(cv));
Xtest  = X(test(cv),:);     Ytest  = Y(test(cv));

% Normalize
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1); sigma(sigma==0)=1;
XtrainZ = (Xtrain - mu) ./ sigma;
XtestZ  = (Xtest  - mu) ./ sigma;

% Train SVM and calibrate posterior
fprintf('Training per-frame SVM...\n');
SVMModel = fitcsvm(XtrainZ, Ytrain, 'KernelFunction','rbf', 'Standardize', false);
SVMPosterior = fitPosterior(SVMModel, XtrainZ, Ytrain);

% Evaluate
[ypred, score] = predict(SVMPosterior, XtestZ);
conf = score(:, strcmp(SVMPosterior.ClassNames, 'Fire'));
acc = mean(ypred == Ytest);
[prec, rec, f1] = precision_recall_f1(Ytest, ypred, 'Fire');
fprintf('Test Acc: %.2f%% | Precision: %.2f | Recall: %.2f | F1: %.2f\n', acc*100, prec, rec, f1);

% Save model
save('per_frame_SVM.mat', 'SVMPosterior', 'mu', 'sigma', 'quantLevels', 'resizeTo');
fprintf('Saved per_frame_SVM.mat\n');
