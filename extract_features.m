function feat = extract_features(frame, quantLevels)
% EXTRACT_FEATURES Extract per-frame features for wildfire detection.
% Inputs:
%   frame - RGB image resized to target size
%   quantLevels - number of gray quantization levels for GLCM-like features
% Output:
%   feat - row vector of features (22 features)

if nargin < 2
    quantLevels = 16; % default quantization if not provided
end

% -------------------- Color Stats (HSV) --------------------
hsvF = rgb2hsv(frame);
H = hsvF(:,:,1); S = hsvF(:,:,2); V = hsvF(:,:,3);
meanH = mean(H(:)); stdH = std(H(:));
meanS = mean(S(:)); stdS = std(S(:));
meanV = mean(V(:)); stdV = std(V(:));

% -------------------- Grayscale --------------------
gray = double(rgb2gray(frame));

% -------------------- Manual GLCM Features --------------------
[glcmContrast, glcmHomog, glcmEnergy, glcmEntropy] = manual_glcm(gray, quantLevels);

% -------------------- DCT Energy (Low-frequency emphasis) --------------------
D = dct2(gray);
sz = min(8, size(D,1));
dctBlock = D(1:sz, 1:sz);
dctEnergy = sum(abs(dctBlock(:)));

% -------------------- Simple HoG-like Edge Histogram --------------------
[Gx, Gy] = gradient(gray);
orient = atan2(Gy, Gx); 
mag = sqrt(Gx.^2 + Gy.^2);

numBins = 8;
angles = linspace(-pi, pi, numBins+1);
hogHist = zeros(1,numBins);
for b = 1:numBins
    mask = orient >= angles(b) & orient < angles(b+1);
    hogHist(b) = sum(mag(mask));
end

[topVal, ~] = max(hogHist);
hogEntropy = -sum((hogHist/sum(hogHist+eps)).*log2((hogHist+eps)/sum(hogHist+eps)));

% -------------------- Pixel Intensity Entropy --------------------
p = imhist(uint8(gray))/numel(gray);
p(p==0) = [];
pixEnt = -sum(p.*log2(p));

% -------------------- Laplacian Variance --------------------
lap = [0 -1 0; -1 4 -1; 0 -1 0];
edgeResp = conv2(gray, lap, 'same'); 
lapVar = var(edgeResp(:));

% -------------------- Combine Features --------------------
feat = [meanH, stdH, meanS, stdS, meanV, stdV, ...
        glcmContrast, glcmHomog, glcmEnergy, glcmEntropy, ...
        dctEnergy, topVal, hogEntropy, pixEnt, lapVar];

% -------------------- Pad to 22 Features --------------------
while numel(feat) < 22
    feat = [feat, 0];
end

% -------------------- Debug (optional) --------------------
% fprintf('Extracted features: ');
% disp(feat);

end
