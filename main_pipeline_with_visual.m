%% MAIN PIPELINE — CPU-OPTIMIZED FEATURE EXTRACTION, VISUALIZATION & TRAINING
clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Deval Jadav
% Project: Early Wildfire Detection Using Smoke Patterns
% Description:
%   CPU-optimized pipeline for wildfire detection using video data.
%   Samples a few frames per video for speed, extracts features, visualizes
%   results, and trains SVM, Random Forest, and KNN.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Dataset Paths and Parameters
dataDir = 'Wildfire_Dataset\subset_B_videos\all_videos';
labelFile = 'Wildfire_Dataset\labels\video_labels.csv';
quantLevels = 16;        % GLCM quantization levels
resizeTo = [112 112];    % Resize frames to speed up processing
framesPerVideo = 5;      % Sample this many frames per video

fprintf("📂 Loading labels...\n");
T = readtable(labelFile);

%% 2. Detect File and Label Columns Automatically
possibleNames = {'VideoName','Filename','File','Name','Video'};
nameVar = intersect(possibleNames, T.Properties.VariableNames);
if isempty(nameVar)
    error("❌ No filename column found in CSV.");
end
videoNames = T.(nameVar{1});

possibleLabels = {'HasSmoke','HasTrue','Label','Smoke','Fire'};
labelVar = intersect(possibleLabels, T.Properties.VariableNames);
if isempty(labelVar)
    error("❌ No label column found in CSV.");
end
labels = T.(labelVar{1});

% Convert labels to logical
if iscell(labels)
    labels = strcmpi(string(labels), 'true') | strcmpi(string(labels), '1') | strcmpi(string(labels), 'yes');
end
labels = logical(labels);

fprintf("✅ Found %d videos in label file.\n", height(T));

%% 3. Precompute feature length
dummyFrame = im2double(zeros(resizeTo(1), resizeTo(2), 3));
sampleFeat = extract_features(dummyFrame, quantLevels);
numFeat = length(sampleFeat);

%% 4. Preallocate
numVideos = numel(videoNames);
X = zeros(numVideos, numFeat*2);  % mean + std per video
Y = labels;

fprintf("🚀 Starting CPU-optimized feature extraction...\n");

%% 5. Feature extraction loop (sample frames only)
for i = 1:numVideos
    vidName = string(videoNames(i));
    vidPath = fullfile(dataDir, vidName);

    if ~isfile(vidPath)
        warning("⚠️ Missing file: %s, skipping.", vidPath);
        continue;
    end

    vr = VideoReader(vidPath);
    totalFrames = floor(vr.FrameRate * vr.Duration);
    idxs = round(linspace(1, totalFrames, min(framesPerVideo, totalFrames)));
    perFrame = zeros(numel(idxs), numFeat);

    for f = 1:numel(idxs)
        vr.CurrentTime = (idxs(f)-1)/vr.FrameRate;
        frame = readFrame(vr);
        frame = imresize(frame, resizeTo);

        try
            feat = extract_features(frame, quantLevels);
            perFrame(f, :) = feat;
        catch ME
            warning("Frame %d in %s skipped: %s", f, vidName, ME.message);
        end
    end

    % Aggregate video-level features (mean + std)
    X(i, :) = [mean(perFrame,1), std(perFrame,0,1)];
    fprintf("[%d/%d] Processed %s\n", i, numVideos, vidName);
end

fprintf("✅ Feature extraction completed for %d videos.\n", size(X,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 6. VISUALIZATION (FAST AND MEANINGFUL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("📊 Visualizing features...\n");

% 6a. Boxplot of first 10 features
figure('Name','Feature Distribution','NumberTitle','off');
boxplot(X(:,1:10));
title('Boxplot of First 10 Features Across All Videos');
xlabel('Feature Index'); ylabel('Value');

% 6b. Mean vs std per feature
meanFeat = mean(X,1);
stdFeat  = std(X,0,1);
figure('Name','Mean vs Std Features','NumberTitle','off');
bar([meanFeat(1:10); stdFeat(1:10)]');
legend({'Mean','Std'});
title('Mean vs Std of First 10 Features');
xlabel('Feature Index'); ylabel('Value');

% 6c. Label distribution
figure('Name','Label Distribution','NumberTitle','off');
histogram(Y, 'FaceColor',[0.2 0.6 0.8]);
title('Video Label Distribution'); ylabel('Count'); xlabel('Class');

% 6d. PCA scatter (2 components)
coeff = pca(X);
Xproj = X*coeff(:,1:2);
figure('Name','PCA Scatter','NumberTitle','off');
gscatter(Xproj(:,1), Xproj(:,2), Y, 'rb', 'o^');
xlabel('PC1'); ylabel('PC2'); title('PCA of Features');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7. Normalize and PCA
mu = mean(X,1);
sigma = std(X,0,1);
sigma(sigma==0)=1;
Xz = (X - mu) ./ sigma;

k = min(25, size(Xz,2));
[coeff, ~, ~] = pca(Xz);
Xpca = Xz * coeff(:,1:k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 8. Train models
fprintf("🤖 Training classifiers...\n");
SVMModel = fitcsvm(Xpca, Y, 'KernelFunction','rbf', 'Standardize', true);
RFModel  = TreeBagger(60, Xpca, Y, 'OOBPrediction','On', 'Method','classification');
KNNModel = fitcknn(Xpca, Y, 'NumNeighbors',5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 9. Save outputs
save('wildfire_pipeline_outputs.mat', 'X','Y','SVMModel','RFModel','KNNModel','mu','sigma','quantLevels','coeff');

fprintf("\n✅ Training complete. Results saved to wildfire_pipeline_outputs.mat\n");
fprintf("------------------------------------------------------------\n");
fprintf("Videos Processed: %d | Feature Length: %d\n", numel(videoNames), numFeat);
fprintf("------------------------------------------------------------\n");
