%% REAL-TIME WILDFIRE DETECTION SIMULATION (CPU-Only)
clc; clear; close all;

%% Load trained model and normalization parameters
load('wildfire_pipeline_outputs.mat', ...
     'SVMModel','mu','sigma','quantLevels','coeff');

dataDir = 'Wildfire_Dataset\subset_B_videos\all_videos'; % Folder with videos
videoList = dir(fullfile(dataDir, '*.mp4'));
numVideos = numel(videoList);

if numVideos == 0
    error("No videos found in folder: %s", dataDir);
end

rng(1);
sel = randperm(numVideos, min(5, numVideos)); % choose up to 5 random videos
resizeTo = [112 112];
windowSize = 5; % number of frames for rolling average
fprintf("Starting real-time simulation on %d random videos...\n", length(sel));

%% Loop through selected videos
for v = 1:length(sel)
    vidName = videoList(sel(v)).name;
    vidPath = fullfile(dataDir, vidName);
    fprintf("Playing video: %s\n", vidName);

    vr = VideoReader(vidPath);
    hFig = figure('Name', sprintf('Wildfire Detection - %s', vidName), ...
                  'NumberTitle','off','Color','w');
    hAx1 = subplot(1,2,1); title(hAx1,'Frame');
    hAx2 = subplot(1,2,2); title(hAx2,'Confidence (Smoke)');
    confPlot = animatedline('Parent',hAx2, 'Color','r','LineWidth',2);
    ylim(hAx2,[0 1]); xlim(hAx2,[0 inf]);
    xlabel(hAx2,'Frame'); ylabel(hAx2,'Confidence');

    frameCount = 0;
    featureBuffer = [];
    tic;

    %% Process video frames
    while hasFrame(vr)
        frame = readFrame(vr);
        frame = imresize(frame, resizeTo);
        frameCount = frameCount + 1;

        % --- Extract per-frame features ---
        feat = extract_features(frame, quantLevels);

        % Ensure correct feature vector length
        if isempty(feat)
            continue;
        elseif length(feat) < length(mu)/2
            % Pad features if needed
            feat = [feat, zeros(1, length(mu)/2 - length(feat))];
        elseif length(feat) > length(mu)/2
            % Truncate if too long
            feat = feat(1:length(mu)/2);
        end

        % --- Update rolling buffer ---
        featureBuffer = [featureBuffer; feat];
        if size(featureBuffer,1) > windowSize
            featureBuffer(1,:) = [];
        end

        % --- Aggregate mean + std (same as training) ---
        meanFeat = mean(featureBuffer,1);
        stdFeat  = std(featureBuffer,0,1);
        aggFeat = [meanFeat, stdFeat];

        % --- Normalize and apply PCA ---
        aggFeatZ = (aggFeat - mu) ./ sigma;
        framePCA = aggFeatZ * coeff(:,1:25);

        % --- Predict smoke/fire ---
        [label, score] = predict(SVMModel, framePCA);

        % --- Extract confidence for positive class ---
        if iscell(SVMModel.ClassNames)
            fireIdx = find(strcmp(SVMModel.ClassNames, '1') | strcmpi(SVMModel.ClassNames, 'true'));
            if isempty(fireIdx), fireIdx = 2; end
        else
            fireIdx = 2;
        end
        conf = score(:, fireIdx);

        % --- Visualization ---
        imshow(frame, 'Parent', hAx1);
        if label
            title(hAx1, sprintf('🔥 Smoke Detected | Frame %d | Conf: %.2f', frameCount, conf), 'Color','r');
        else
            title(hAx1, sprintf('No Smoke | Frame %d | Conf: %.2f', frameCount, conf), 'Color','k');
        end
        addpoints(confPlot, frameCount, conf);

        if mod(frameCount, 5) == 0
            drawnow limitrate;
        end
    end

    fprintf("Finished %s in %.2f sec\n", vidName, toc);
end

fprintf("✅ Real-time simulation complete.\n");
