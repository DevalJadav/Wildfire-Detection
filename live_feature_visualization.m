%% LIVE SMOKE DETECTION - CPU (SIDE-BY-SIDE)
clc; clear; close all;

% ====================== PARAMETERS ======================
dataDir = 'Wildfire_Dataset\subset_B_videos\all_videos';
resizeTo = [112 112];
quantLevels = 16;
expectedFeatures = 25;  % your SVM expects 25 features

% Load trained SVM
try
    load('wildfire_pipeline_outputs.mat', 'SVMModel');
catch ME
    error('Failed to load SVM model: %s', ME.message);
end

% ====================== RANDOM VIDEO ======================
videoFiles = dir(fullfile(dataDir, '*.mp4'));
if isempty(videoFiles)
    error('No videos found in %s', dataDir);
end

vidChoice = videoFiles(randi(length(videoFiles))).name;
fprintf("🎬 Random video selected: %s\n", vidChoice);

vr = VideoReader(fullfile(dataDir, vidChoice));
totalFrames = floor(vr.FrameRate * vr.Duration);

% ====================== VIDEO DISPLAY ======================
figure('Name','Live Smoke Detection - Side by Side','NumberTitle','off');

for f = 1:totalFrames
    try
        frame = readFrame(vr);

        % Skip if frame too small
        if size(frame,1) < resizeTo(1) || size(frame,2) < resizeTo(2)
            warning('Frame %d too small, skipped', f);
            continue;
        end

        % Resize frame
        frameResized = imresize(frame, resizeTo);

        % Grayscale for smoke visualization
        grayFrame = rgb2gray(frameResized);

        % ====================== FEATURE EXTRACTION ======================
        feat = extract_features(frameResized, quantLevels);

        % Ensure exactly expectedFeatures
        if numel(feat) < expectedFeatures
            feat(end+1:expectedFeatures) = 0;   % pad with zeros
        elseif numel(feat) > expectedFeatures
            feat = feat(1:expectedFeatures);    % truncate
        end

        % Normalize
        featZ = (feat - mean(feat)) ./ (std(feat) + eps);

        % ====================== PREDICTION ======================
        smokePred = predict(SVMModel, featZ);

        % ====================== HIGHLIGHT SMOKE ======================
        overlay = repmat(grayFrame,1,1,3);  % convert to RGB

        if smokePred
            % Create smoke mask based on intensity
            smokeMask = grayFrame > 120 & grayFrame < 200;
            smokeMask = double(smokeMask);
            overlay(:,:,1) = uint8(min(double(overlay(:,:,1)) + 100*smokeMask, 255));
        end

        % ====================== SIDE-BY-SIDE DISPLAY ======================
        combinedFrame = cat(2, frameResized, overlay);  % original | highlighted
        imshow(combinedFrame);
        title(sprintf('Frame %d/%d - Smoke: %s', f, totalFrames, string(smokePred)));
        drawnow;

    catch ME
        warning(ME.identifier, 'Frame %d skipped: %s', f, ME.message);
        continue;
    end
end
