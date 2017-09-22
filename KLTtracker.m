function [pointsList] = KLTtracker(images, xiTr, yiTr)
% KLTtracker
first_frame = images(:,:,1);
numFrame = size(images, 3);


% % debugging:
% images = vg;
% 
% xiTr = xoutTr;
% yiTr = youtTr;
% xi2 = xout2;
% yi2 = yout2;
% clear xoutTr youtTr xout2 yout2
xmaxTr = max(xiTr);
xminTr = min(xiTr);
ymaxTr = max(yiTr);
yminTr = min(yiTr);

xmax2 = xmaxTr;
xmin2 = xminTr;
ymax2 = ymaxTr;
ymin2 = yminTr;

wTr = xmaxTr - xminTr;
hTr = ymaxTr - yminTr;

w2 = xmax2 - xmin2;
h2 = ymax2 - ymin2;

ROIpointsTr = [xiTr, yiTr];
ROIboxTr = [xminTr yminTr wTr hTr]; % ROI enclosed by xi and yi

ROIpoints2 = ROIpointsTr;% [xi2, yi2];
ROIbox2 = ROIboxTr;%[xmin2 ymin2 w2 h2];
% initialize grid points, get each pixel within the ROI, counter should = w*h
counter = 0;
for i = 1:w2
    for j = 1:h2
        pixelX = xmin2 + i;  
        pixelY = ymin2 + j;
        counter = counter+1;
%         if regionMask(pixelX,pixelY)
            pointsList_init(counter, 1) = pixelX;
            pointsList_init(counter, 2) = pixelY;
%         end
    end
end

% % check where the grid points are
% testImg = first_frame;
% testImg = insertMarker(testImg, pointsList_init, '+', 'Color', 'blue');
% figure, imshow(testImg)

% detect feature poitns
pointsStr = detectMinEigenFeatures(first_frame, 'MinQuality', 0.001, 'ROI', ROIboxTr);

% select the strongest K points 
numKLT = round(0.85*size(pointsStr,1));
pointsStr = pointsStr.selectStrongest(numKLT); 

% reject points outside the polygon 
in_sel = inpolygon(pointsStr.Location(:,1), pointsStr.Location(:,2),xiTr,yiTr); 
points = pointsStr.Location(in_sel,:); 

% displayImage = first_frame;
% displayImage = insertShape(displayImage, 'Rectangle', ROIboxTr, 'Color', 'red');
% displayImage = insertMarker(displayImage, points, '+', 'Color', 'white');
% figure, imshow(displayImage);


%%

% create a tracker object
tracker = vision.PointTracker('MaxBidirectionalError', 1);
% initialize the tracker
initialize(tracker, points, first_frame);
% psize = size(pointsList_init);
% pointsList = zeros(psize(1), 2, numFrame);
pointsList = zeros(numFrame, size(pointsList_init, 1), 2);
pointsListpoints = pointsList_init;
oldPoints = points;
% track the points:
for i = 1:numFrame

%             if (rem(i,100)<1)
%                 disp(['percentage complete = ',num2str(i*100/numFrame)])
%             end
      next_frame = images(:,:,i);
      [points, isFound] = step(tracker, images(:,:,i));
      visiblePoints = points(isFound, :);
      oldInliers = oldPoints(isFound, :);
      
      if size(visiblePoints, 1) >= 20 % need at least 2 points
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
           [xform] = estimateGeometricTransform(...
           oldInliers, visiblePoints, 'similarity');%, 'MaxDistance', 2);
        % Apply the transformation to the bounding box points
        pointsListpoints = transformPointsForward(xform, pointsListpoints);
        xform.T;
        % Display tracked points
%         next_frame = insertMarker(next_frame, visiblePoints, '+', ...
%             'Color', 'white');
%         imshow(next_frame);
%         
        pointsList(i,:,:) = pointsListpoints;
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(tracker, oldPoints);
        i;
       
      else
          disp('not tracking')
      end
      
      if round(permute(pointsList(i,:,:), [2 3 1])) == round(pointsList_init(:,:))
%           disp('All same...')
      end
end
        % Clean up
        release(tracker);
        
%         % check how the ROI was transformed
%         testROI = images(:,:,numFrame); % last frame
% %         testROI = insertShape(testROI, 'Rectangle', ROIboxTr, 'Color', 'red');
%         testROI = insertMarker(testROI, permute(pointsList(numFrame, :,:), [2 3 1]), '+', 'Color', 'blue');
%         testROI = insertMarker(testROI, pointsList_init, '+', 'Color', 'yellow');
% 
%        figure, imshow(testROI);

% end 
