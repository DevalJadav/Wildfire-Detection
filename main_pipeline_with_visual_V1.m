%% MAIN PIPELINE — CPU-OPTIMIZED FEATURE EXTRACTION, VISUALIZATION & TRAINING
clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Deval Jadav
% Project: Early Wildfire Detection Using Smoke Patterns
% Description:
%   CPU-optimized pipeline for wildfire detection using video data.
%   Includes feature extraction, live visualization, PCA, and ML model training.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Dataset Paths and Parameters
dataDir = 'Wildfire_Dataset\subset_B_videos\all_videos';
labelFile = 'Wildfire_Dataset\labels\video_labels.csv';
quantLevels = 16;        % GLCM quantization levels
resizeTo = [112 112];    % Resize dimension
framesPerVideo = 5;      % Sample frames per video

%% 2. Load labels
fprintf("📂 Loading labels...\n");
T = readtable(labelFile);

% Detect file and label columns automatically
possibleNames = {'VideoName','Filename','File','Name','Video'};
nameVar = intersect(possibleNames, T.Properties.VariableNames);
videoNames = T.(nameVar{1});

possibleLabels = {'HasSmoke','HasTrue','Label','Smoke','Fire'};
labelVar = intersect(possibleLabels, T.Properties.VariableNames);
labels = T.(labelVar{1});
if iscell(labels)
    labels = strcmpi(string(labels),'true') | strcmpi(string(labels),'1') | strcmpi(string(labels),'yes');
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

%% 5. Live visualization setup
figure('Name','Live Feature Monitoring','NumberTitle','off');
hBar = bar(nan(1,numFeat),'FaceColor',[0.2 0.6 0.8]);
xlabel('Feature Index'); ylabel('Mean Feature Value');
title('Live Video Feature Monitoring');

fprintf("🚀 Starting CPU-optimized feature extraction...\n");

%% 6. Feature extraction loop (sample frames only)
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
            perFrame(f,:) = feat;
        catch ME
            warning("Frame %d in %s skipped: %s", f, vidName, ME.message);
        end
    end

    % Aggregate video-level features (mean + std)
    X(i,:) = [mean(perFrame,1), std(perFrame,0,1)];

    % --- LIVE UPDATE ---
    hBar.YData = mean(perFrame,1);
    title(sprintf('Processing [%d/%d]: %s', i, numVideos, vidName));
    drawnow;
end

fprintf("✅ Feature extraction completed for %d videos.\n", size(X,1));

%% 7. Visualization after feature extraction
fprintf("📊 Visualizing features...\n");

% 7a. Boxplot of first 10 features
figure('Name','Feature Distribution','NumberTitle','off');
boxplot(X(:,1:10));
title('Boxplot of First 10 Features Across Videos');
xlabel('Feature Index'); ylabel('Value');

% 7b. Mean vs Std
meanFeat = mean(X,1);
stdFeat = std(X,0,1);
figure('Name','Mean vs Std Features','NumberTitle','off');
bar([meanFeat(1:10); stdFeat(1:10)]');
legend({'Mean','Std'}); xlabel('Feature Index'); ylabel('Value');
title('Mean vs Std of First 10 Features');

% 7c. Label distribution
figure('Name','Label Distribution','NumberTitle','off');
histogram(Y, 'FaceColor',[0.2 0.6 0.8]);
xlabel('Class'); ylabel('Count'); title('Video Label Distribution');

% 7d. Feature correlation (first 12)
figure('Name','Feature Correlation','NumberTitle','off');
corrMat = corr(X(:,1:12));
imagesc(corrMat); colorbar; xticks(1:12); yticks(1:12);
title('Correlation Matrix of First 12 Features');

% 7e. PCA scatter
coeff = pca(X);
Xproj = X*coeff(:,1:2);
figure('Name','PCA Scatter','NumberTitle','off');
gscatter(Xproj(:,1), Xproj(:,2), Y,'rb','o^');
xlabel('PC1'); ylabel('PC2'); title('PCA of Features');

%% 8. Normalize & PCA for training
mu = mean(X,1); sigma = std(X,0,1); sigma(sigma==0)=1;
Xz = (X-mu)./sigma;

k = min(25, size(Xz,2));
[coeff, ~, ~] = pca(Xz);
Xpca = Xz*coeff(:,1:k);

%% 9. Train models
fprintf("🤖 Training classifiers...\n");
SVMModel = fitcsvm(Xpca,Y,'KernelFunction','rbf','Standardize',true);
RFModel = TreeBagger(60,Xpca,Y,'OOBPrediction','On','Method','classification');
KNNModel = fitcknn(Xpca,Y,'NumNeighbors',5);

%% 10. Save outputs
save('wildfire_pipeline_outputs.mat', 'X','Y','SVMModel','RFModel','KNNModel','mu','sigma','quantLevels','coeff');

fprintf("\n✅ Pipeline complete. Results saved.\n");
fprintf("------------------------------------------------------------\n");
fprintf("Videos Processed: %d | Feature Length: %d\n", numVideos, numFeat);
fprintf("------------------------------------------------------------\n");
