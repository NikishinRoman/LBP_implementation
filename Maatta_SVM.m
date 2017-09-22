% load saved LBP patterns for testing and training data 
% initialize variables for saving
    predictionAllSVM = [];
    predictionAllSVMLive = [];
    predictionAllSVMFake = [];
    testPeople = [];
    labelsSVM = [];
    orderTrAll = [];
    orderTsAll = []; 
%     ii =1; % if only one foldr in the list
    Folder = 'Masks_LBP_singleFrame/' ;%[folderList{ii} '/'];       % folder with LBP  data
% per person - get a list of training and testing people
% initialize LOOV
% if replay_mask == 0  % replay data
%     liveFolders = 1;
%     fakeFolders = 2;    
%     startTestPerson1 = (1:4:(60));
%     startTestPerson2 = (1:4:(60))+60;
%     startTestPerson3 = (1:2:(60))+120 ;
%     
%     endTestPerson1 =   ((1:4:(60)) + 3 );
%     endTestPerson2 =     (((1:4:(60))+60) +3);
%     endTestPerson3 =     (((1:2:(60))+120) +1); 
%     
%     allPeople = startTestPerson1(1):endTestPerson3(end);
%     pEnd = 15;
% elseif replay_mask == 1
    liveFolders = 2;
    fakeFolders = 3;
%     startTestPerson1 = [1:5:85];
%     endTestPerson1 = [1:5:85] + 4;

    startTestPerson2 = [];
    startTestPerson3 = [];
    endTestPerson2 = [];
    endTestPerson3 = [];

    allPeople = startTestPerson1(1):endTestPerson1(end); % ALL  %startPerson(1):startPerson(p);
%     pEnd = 17;
% else
%     disp('ERROR')
% end
   for p = 1:pEnd;  % p = 1:17 if 3DMAD, 15 per folder for Replay - test, train, devel - different people
%     p=1;
    testPerson1 = startTestPerson1(p):endTestPerson1(p);
    if isempty(startTestPerson2) ~= 1
        testPerson2 = startTestPerson2(p):endTestPerson2(p);
    else
        testPerson2 = [];
    end
     if isempty(startTestPerson2) ~= 1
        testPerson3 = startTestPerson3(p):endTestPerson3(p);
     else
         testPerson3 = [];
     end
    testPerson = [testPerson1 testPerson2 testPerson3];

    trainPeople = setdiff(allPeople, testPerson);

% read in data as trainingLive, TestingLive, TrainingFake and TestingFake
PdataLtr = [];
          for f = liveFolders
            Pi = [];
            for ff = 1:length(trainPeople)
                try
            m = trainPeople(ff);
            load([Folder num2str(f) '-' num2str(m) '-LBPdata'])
            Pi = LBP_finalVec;
            if sum(sum(isnan(Pi)))
                continue
            end
            PdataLtr = [PdataLtr; Pi];
                catch
                continue
                end
            
            end   
          end
 PdataLts = [];
          for f = liveFolders
              Pi = [];
            for ff = 1:length(testPerson)
                try
            m = testPerson(ff);            
            load([Folder num2str(f) '-' num2str(m) '-LBPdata'])
            Pi = LBP_finalVec;
            if sum(sum(isnan(Pi)))
                continue
            end
             PdataLts = [PdataLts; Pi];
             catch
            continue
            end
            end   
          end
          
     PdataFtr = [];
          for f = fakeFolders
              Pi = [];
            for ff = 1:length(trainPeople)
                try
            m = trainPeople(ff);
            load([Folder num2str(f) '-' num2str(m) '-LBPdata'])
            Pi = LBP_finalVec;
            if sum(sum(isnan(Pi)))
                continue
            end
             PdataFtr = [PdataFtr; Pi];
             catch
            continue
            end
            end   
          end
          
 PdataFts = [];
          for f = fakeFolders
              Pi = [];
            for ff = 1:length(testPerson)
                try
            m = testPerson(ff);
            load([Folder num2str(f) '-' num2str(m) '-LBPdata'])
            Pi = LBP_finalVec;
            if sum(sum(isnan(Pi)))
                continue
            end
             PdataFts = [PdataFts; Pi];
             catch
            continue
            end
            end   
          end
            
            
            
            
%% prepare data for SVM
% if isempty(PdataLtr) || isempty(PdataFtr) || isempty(PdataLts) || isempty(PdataFts)
%             break % skip if there is data missing! otherwise - overfits
%         end
        YtrL = ones(size(PdataLtr,1), 1);
        YtrF = zeros(size(PdataFtr,1), 1);
        YtsL = ones(size(PdataLts,1), 1);
        YtsF = zeros(size(PdataFts,1), 1);

        % combine. No shuffling if LOOV?
        YtrTemp = [YtrL; YtrF];
        YtsTemp = [YtsL; YtsF];

        XtrTemp = [PdataLtr; PdataFtr];
        XtsTemp = [PdataLts; PdataFts];

        XYtrTemp = [XtrTemp YtrTemp];
        s = RandStream('mt19937ar','Seed',sum(100*clock));
        orderTri = randperm(s, size(XYtrTemp,1));
        XYtr = XYtrTemp(orderTri,:);

        Xtr = XYtr(:,1:(end-1));
        Ytr = XYtr(:,end);

        XYtsTemp = [XtsTemp YtsTemp];
        s = RandStream('mt19937ar','Seed',sum(100*clock));
        orderTsi = randperm(s, size(XYtsTemp,1));
        XYts = XYtsTemp(orderTsi,:);

        Xtr = XYtr(:,1:(end-1));
        Ytr = XYtr(:,end);

        Xts = XYts(:,1:(end-1));
        Yts = XYts(:,end);
%% SVM
% tr - 1743 real imgs and 1748 imposter images (fake and live of the same person)
% ts - 3362 real and 5761 imposter 

 SVMModel = fitcsvm(Xtr,Ytr,'KernelFunction','linear','Standardize',true);
        [labelSVM,score] = predict(SVMModel,Xts);
        predictionSVM = (length(find(labelSVM==Yts))/length(Yts))*100;
        predictionAllSVM = [predictionAllSVM; predictionSVM];    

        % prediction for live and fake separately 
        LiveIdx = find(Yts == 1);
        FakeIdx = find(Yts == 0);
        labelLive = labelSVM(LiveIdx); %label(1:end/2);
        labelFake = labelSVM(FakeIdx); %label((end/2+1):end);
        YtsLive = Yts(LiveIdx); %Yts(1:end/2);
        YtsFake = Yts(FakeIdx); %Yts((end/2+1):end);

        predictionSVMLive = (length(find(labelLive==YtsLive))/length(YtsLive))*100;
        predictionAllSVMLive = [predictionAllSVMLive; predictionSVMLive]; 
        predictionSVMFake = (length(find(labelFake==YtsFake))/length(YtsFake))*100;
        predictionAllSVMFake = [predictionAllSVMFake; predictionSVMFake]; 
        
        scores_SVMcell{p} = num2cell(score);
        Ytsscell{p} = Yts;
        Ytrscell{p} = Ytr;
        testPeople = [testPeople; testPerson];
        labelsSVMcell{p} = labelSVM;
        orderTscell{p} = orderTsi;
  end
%% Error computation
% my metric that i used= total % misclassified for live and for fake
        predictionAverageSVM = sum(predictionAllSVM)/length(predictionAllSVM);
        disp([num2str(predictionAverageSVM) '% Average SVM accuracy']);

        predictionAverageSVMLive = sum(predictionAllSVMLive)/length(predictionAllSVMLive);
        disp([num2str(predictionAverageSVMLive) '% Average Live SVM accuracy']);
        predictionAverageSVMFake = sum(predictionAllSVMFake)/length(predictionAllSVMFake);
        disp([num2str(predictionAverageSVMFake) '% Average Fake SVM accuracy']);
        