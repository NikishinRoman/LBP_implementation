%implement multiscale LBP
%% read in videos
% initialize
SaveFolder = 'Replay_LBPTrain';
mkdir(SaveFolder)
% folderMain= '/media/ewa/SH/3dmadDirectories/';
% for f = 2:3
% folderEnd = ['Data0' num2str(f) 'Keep/'];

folderMain = '/media/ewa/SH/ReplayAttackDirectories/';
folderReal = 'train/real/';
folderAttackf = 'train/attack/fixed/';  % both contain photo, vid in adverse and controlled
folderAttackh = 'train/attack/hand/';
savefolder = 'Nov23NotNormReplay';

for f = 3% 1%:2
    if f == 1
        folderEnd = folderReal;
    end
    if f == 2
        folderEnd = folderAttackf;

    end
    if f == 3
        folderEnd = folderAttackh;
    end
fileNameList = dir([[folderMain folderEnd] [ '*.mov']]); 
    for i =1:length(fileNameList)
        imgCells{i} = fileNameList(i).name;  
    end
  [cs,index] = sort_nat(imgCells,'ascend');
    img_names = cs;
   for m = 42:length(fileNameList)
       try
            vidName = img_names{m};       
            % read in the videos
            v = VideoReader([folderMain folderEnd vidName]);
            videoLength = v.Duration;
            videoRate = v.FrameRate;
            numFrame = videoLength*videoRate;
            width = v.Width;
            height = v.Height; 
            frames = read(v);
     load(['../Data/IDAP1stFrameNew/' num2str(f) 'Dlib/dLib-' vidName '.png.mat'])
%     load(['../Data/3DMAD1stFrame/Data0' num2str(f) 'Dlib/dLib-' vidName(1:end-4) '_C.avi' '.png.mat'])
    firstPoints = pointsResized;

%% track to get a face ROI for the whole video
% initialize FaceROI
% they don't say which face detection they use
minFaceW_i = min(firstPoints(:,1));
maxFaceW_i = max(firstPoints(:,1));
minFaceH_i = min(firstPoints(:,2));
maxFaceH_i = max(firstPoints(:,2));

% faceW_i = maxFaceW_i - minFaceW_i;
% faceH_i = maxFaceH_i - minFaceH_i;

xiTr = [minFaceW_i; maxFaceW_i; maxFaceW_i; minFaceW_i; minFaceW_i];
yiTr = [minFaceH_i; minFaceH_i; maxFaceH_i; maxFaceH_i; minFaceH_i];
% klt needs it to only have one channel?
framesGreen = frames(:,:,2,:);
framesGreen = permute(framesGreen, [1 2 4 3]);

pointsList = KLTtracker(framesGreen,xiTr, yiTr);
LBP_finalVec = [];
%% grayscale img
 for ii = 1:size(frames,4) % before we can run on the whole image, need to apply tracking ...
            %for facial points or use a different face detector
% convert each video to a grayscale image 
img = frames(:,:,:,ii);  % take one frame and convert to gray
imgG_i = rgb2gray(img);
%% crop face ROI and resize
minFaceW = min(pointsList(ii,:,1));
maxFaceW = max(pointsList(ii,:,1));
minFaceH = min(pointsList(ii,:,2));
maxFaceH = max(pointsList(ii,:,2));

faceW = maxFaceW - minFaceW;
faceH = maxFaceH - minFaceH;

I_1 = imcrop(imgG_i, [minFaceW minFaceH faceW faceH]);
I = imresize(I_1, [64 64]);  % stretch aspect ratio???
%% global LBP
% for each image
% get a resulting global LBP face image 
R1 = 1;
N1 = 8;
MAPPING1=getmapping(8,'u2');
MODEimg = i;
% how to make it output an LBP Image instead ogf a feature vector - use i
% for MODE
LBP_img = lbp(I,R1,N1,MAPPING1,MODEimg);

MODE = 'hist';
H_u2_8_1 = [];
% counter = 0;
% divide into overlapping 3x3 regions - LBP_img or I?
% [row col]=size(LBP_img);
% for i=1:3:row-2
%      for j = 1:3:col-2
%        I_3x3=LBP_img(i:(i+2), j:(j+2));  % 3x3 overlap regions inside the LBP image
%      LBP_img3x3 = lbp(I_3x3,R1,N1,MAPPING1,MODE);
% % each region should give 59-bin histogram - combine to end up with
% % 531-bin, 9 3x3 ROIs
%  % concatenate
%  H_u2_8_1 = [H_u2_8_1 LBP_img3x3];        % wrong length
% counter = counter +1;
%     end
% end

% makes counter = 9?
[row col]=size(LBP_img);
counter = 0;
for i=10:20:row-9
     for j = 10:20:col-9
       I_3x3=LBP_img(j-9:j+9,i-9:i+9)  ;     
       LBP_img3x3 = lbp(I_3x3,R1,N1,MAPPING1,MODE);
        H_u2_8_1 = [H_u2_8_1 LBP_img3x3];
       counter = counter +1;
    end
end

% repeat LBP but on the whole face ROI
N2 = 8;
R2 = 2;
H_u2_8_2 = lbp(I,R2,N2,MAPPING1,MODE);

MAPPING2=getmapping(16,'u2');
N3 = 16;
H_u2_16_2 = lbp(I,R2,N3,MAPPING2,MODE);

LBP_finalVeci = [H_u2_8_1 H_u2_8_2 H_u2_16_2]; % does the order matter?
LBP_finalVec = [LBP_finalVec; LBP_finalVeci];
% feature vector for each observation (img)
% total number of histogram bins should be 833 = 531+59+243
% if f ==2 || f==1   % tr
%     XtrTemp = [XtrTemp; LBP_finalVec];
% elseif f==3
%     XtsTemp = [XtsTemp; LBP_finalVec];
% end
    end
       catch 
           continue
       end
       %% save LBP patterns for training and testing data
   save([SaveFolder '/' num2str(f) '-' num2str(m) '-' 'LBPdata.mat'], 'LBP_finalVec')
 
      disp([num2str(f) '-' num2str(m)])
 end
   end
 
