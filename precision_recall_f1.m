function [precision, recall, f1] = precision_recall_f1(Ytrue,Ypred,posClass)
Yt = Ytrue==posClass;
Yp = Ypred==posClass;

TP = sum(Yt & Yp);
FP = sum(~Yt & Yp);
FN = sum(Yt & ~Yp);

precision = TP/(TP+FP+eps);
recall = TP/(TP+FN+eps);
f1 = 2*(precision*recall)/(precision+recall+eps);
end
