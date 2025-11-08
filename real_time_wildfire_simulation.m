%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REAL-TIME WILDFIRE DETECTION SIMULATION (PER-FRAME SVM)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% --- Load trained per-frame SVM ---
fprintf('Loading trained per-frame SVM...\n');
load('per_frame_SVM.mat', 'SVMPosterior', 'mu', 'sigma', 'quantLevels');

%% --- Video setup ---
videoPath = 'Wildfire_Dataset\subset_B_videos\all_videos\heinola_1.mp4';
vidObj = VideoReader(videoPath);
resizeTo = [224 224];

frameCount = 0;
prevFeat = [];

%% --- Visualization setup ---
figure('Name','Wildfire Detection Simulation','NumberTitle','off');
subplot(2,1,1);
frameDisplay = imshow(zeros([resizeTo 3],'uint8'));
title('Video Frame');

subplot(2,1,2);
confPlot = animatedline('Color','r','LineWidth',2);
ylim([0 1]); grid on;
xlabel('Frame'); ylabel('Fire Confidence');
title('Fire Detection Confidence (SVM)');

%% --- Process frames ---
fprintf('Running simulation on video: %s\n', videoPath);

while hasFrame(vidObj)
    frameCount = frameCount + 1;
    frame = readFrame(vidObj);
    frame = imresize(frame, resizeTo);

    % --- Extract features ---
    feat = extract_features(frame, quantLevels);

    % --- Delta from previous frame ---
    if isempty(prevFeat)
        deltaFeat = zeros(size(feat));
    else
        deltaFeat = feat - prevFeat;
    end
    prevFeat = feat;

    % --- Combined feature vector ---
    Xnew = [feat, deltaFeat];

    % --- Normalize ---
    if length(Xnew) ~= length(mu)
        error('Feature length mismatch! Xnew=%d, mu=%d', length(Xnew), length(mu));
    end
    XnewZ = (Xnew - mu) ./ sigma;

    % --- Predict probability ---
    [~, score] = predict(SVMPosterior, XnewZ);

    % Safe indexing for 'Fire'
    if iscell(SVMPosterior.ClassNames)
        fireIdx = find(strcmp(SVMPosterior.ClassNames, 'Fire'));
    else
        fireIdx = find(SVMPosterior.ClassNames == 'Fire');
    end
    if isempty(fireIdx)
        conf = 0;
    else
        conf = score(:, fireIdx);
    end

    % --- Visualization ---
    set(frameDisplay, 'CData', frame);
    addpoints(confPlot, frameCount, conf);
    drawnow limitrate;

    % --- Alert ---
    if conf > 0.7
        sgtitle(sprintf('🔥 Fire Detected! Confidence: %.2f', conf), 'Color', 'r');
    else
        sgtitle(sprintf('No Fire (Conf: %.2f)', conf), 'Color', 'k');
    end
end

fprintf('Simulation completed on %d frames.\n', frameCount);
