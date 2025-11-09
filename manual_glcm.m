function [contrast, homogeneity, energy, entropyVal] = manual_glcm(gray, levels)
gray = double(gray);
gmin = min(gray(:)); gmax = max(gray(:));
if gmax==gmin
    q = zeros(size(gray));
else
    q = floor((gray-gmin)/(gmax-gmin+eps)*(levels-1));
end
q = uint8(q);

GLCM = zeros(levels,levels);
[r,c] = size(q);
for i=1:r
    for j=1:c-1
        GLCM(double(q(i,j))+1,double(q(i,j+1))+1) = GLCM(double(q(i,j))+1,double(q(i,j+1))+1)+1;
    end
end

GLCM = GLCM / sum(GLCM(:)) + eps;

contrast=0; homogeneity=0; energy=0; entropyVal=0;
for a=1:levels
    for b=1:levels
        contrast = contrast + (abs(a-b)^2)*GLCM(a,b);
        homogeneity = homogeneity + GLCM(a,b)/(1+abs(a-b));
        energy = energy + GLCM(a,b)^2;
        entropyVal = entropyVal - GLCM(a,b)*log2(GLCM(a,b));
    end
end
end
