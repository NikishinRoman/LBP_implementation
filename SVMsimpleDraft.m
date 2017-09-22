% SVM simple try
%% prepare data for SVM
% save the LBP vectors into an observation-feature matrix, separately for
% XtrTemp and XtsTemp

% add labels
YtrTemp = ones(size(XtrTemp,1),1);
YtsTemp = zeros(size(XtsTemp,1),1);

% shuffle, keep the shuffle order 
XYTrTemp = [XtrTemp YtrTemp];
XYTsTemp = [XtsTemp YtsTemp];

XYTemp = [XYTrTemp; XYTsTemp];
s = RandStream('mt19937ar','Seed',sum(100*clock));
% train
orderi = randperm(s, size(XYTemp,1));
XY = XYTemp(orderi,:);
% keep the order after shuffling
[~,order] = sort(orderi);    % keep to unshuffle
% XYtr_ordered = XYtr(orderTr, :);

% split into X and Y - tr and ts
% do LOOV later instead
Xtr = XY(1:20,1:end-1);
Xts = XY(21:end,1:end-1);
Ytr = XY(1:20,end);
Yts = XY(21:end,end);

%% SVM
% tr - 1743 real imgs and 1748 imposter images (fake and live of the same person)
% ts - 3362 real and 5761 imposter 

SVMModel = fitcsvm(Xtr,Ytr,'KernelFunction','rbf','Standardize',true);

[label,score] = predict(SVMModel,Xts);
predictionSVM = (length(find(label==Yts))/length(Yts))*100;
% disp([num2str(prediction) '% accuracy']);
% predictionAllSVM = [predictionAllSVM; predictionSVM]; 