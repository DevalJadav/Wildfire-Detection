function feat = extract_features(frame, quantLevels)
    %% Step 1: Convert and normalize input
    frame = im2double(frame);  % convert to [0,1]
    if size(frame,3) == 1
        frame = repmat(frame, [1 1 3]); % ensure RGB
    end

    %% Step 2: Color-based features
    meanRGB = mean(reshape(frame, [], 3));
    stdRGB  = std(reshape(frame, [], 3));

    hsvFrame = rgb2hsv(frame);
    meanHSV = mean(reshape(hsvFrame, [], 3));
    stdHSV  = std(reshape(hsvFrame, [], 3));

    %% Step 3: Texture features (GLCM)
    grayFrame = im2uint8(rgb2gray(frame));
    glcm = zeros(quantLevels, quantLevels);
    step = floor(256 / quantLevels);
    for i = 1:size(grayFrame,1)
        for j = 1:size(grayFrame,2)-1
            r = floor(double(grayFrame(i,j))/step) + 1;
            c = floor(double(grayFrame(i,j+1))/step) + 1;
            r = min(r, quantLevels); c = min(c, quantLevels);
            glcm(r,c) = glcm(r,c) + 1;
        end
    end
    glcm = glcm / sum(glcm(:)) + eps;

    [x,y] = meshgrid(1:quantLevels,1:quantLevels);
    contrast = sum(sum((x-y).^2 .* glcm));
    energy = sum(glcm(:).^2);
    homogeneity = sum(sum(glcm ./ (1 + abs(x-y))));
    mu_x = sum(sum(x .* glcm)); mu_y = sum(sum(y .* glcm));
    sigma_x = sqrt(sum(sum((x - mu_x).^2 .* glcm)));
    sigma_y = sqrt(sum(sum((y - mu_y).^2 .* glcm)));
    corr = sum(sum(((x - mu_x).*(y - mu_y).*glcm))) / (sigma_x*sigma_y + eps);
    texFeat = [contrast, corr, energy, homogeneity];

    %% Step 4: Frequency features
    F = fft2(double(grayFrame));
    freqEnergy = sum(abs(F(:)).^2) / numel(F);

    %% Step 5: Edge features
    lap = [0 -1 0; -1 4 -1; 0 -1 0];
    edgeResp = conv2(double(grayFrame), lap, 'same');
    edgeVar = var(edgeResp(:));

    %% Step 6: Combine features
    feat = [meanRGB, stdRGB, meanHSV, stdHSV, texFeat, freqEnergy, edgeVar];

    %% Step 7: Clean feature vector
    feat(~isfinite(feat)) = 0;
end
