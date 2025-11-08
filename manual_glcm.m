function [contrast, homogeneity, energy, entropyVal] = manual_glcm(gray, levels)
% MANUAL_GLCM Compute a simple gray-level co-occurrence matrix (offset [0 1])
% gray: double grayscale image (0-255 or 0-1)
% levels: number of quantization levels (e.g., 16)
% Returns contrast, homogeneity, energy, entropy

if nargin < 2
    levels = 16;
end

% Ensure gray is in 0..1
if max(gray(:)) > 1
    gray = gray / 255;
end

% 1) Quantize gray into levels 0..levels-1
gmin = min(gray(:)); gmax = max(gray(:));
if gmax == gmin
    q = zeros(size(gray));
else
    q = floor((gray - gmin) / (gmax - gmin + eps) * (levels-1));
end
q = uint8(q);

% 2) Build GLCM for offset (0,1) (right neighbor)
GLCM = zeros(levels, levels);
[r,c] = size(q);
for i = 1:r
    for j = 1:c-1
        rowIdx = double(q(i,j)) + 1;
        colIdx = double(q(i,j+1)) + 1;
        GLCM(rowIdx, colIdx) = GLCM(rowIdx, colIdx) + 1;
    end
end

% 3) Normalize GLCM
GLCM = GLCM / (sum(GLCM(:)) + eps);

% 4) Compute statistics
contrast = 0; homogeneity = 0; energy = 0; entropyVal = 0;
for a = 1:levels
    for b = 1:levels
        p = GLCM(a,b);
        contrast = contrast + (abs(a-b)^2) * p;
        homogeneity = homogeneity + p / (1 + abs(a-b));
        energy = energy + p^2;
        entropyVal = entropyVal - p * log2(p + eps);
    end
end

end
