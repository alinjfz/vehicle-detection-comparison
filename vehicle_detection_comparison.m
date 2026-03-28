% Just setting up the figure size manually
figure('Position', [100, 100, 1200, 700]);

% Load in ground truth data and the video
load('export_ground_truth.mat');
videoName = 'traffic.mj2';
videoReader = VideoReader(videoName);

% usable frames
gtFrameCount = height(gTruth.LabelData);
videoFrameCount = videoReader.NumFrames;
usableFrames = min(gtFrameCount, videoFrameCount);

% ground truth bounding boxes
gtAllBoxes = cell(usableFrames, 1);

% grab the GT boxes
for frameIdx = 1:usableFrames
    rawBox = gTruth.LabelData.Car{frameIdx};
    if isa(rawBox, "struct")
        rawBox = rawBox.Position;
    end
    if isempty(rawBox)
        rawBox = zeros(0,4);
    end
    gtAllBoxes{frameIdx} = rawBox;
end

% the detectors
fprintf('Running Threshold detector\n');
thresholdBoxes = threshold_detector(videoName, usableFrames, gtAllBoxes);

fprintf('Running GMM detector\n');
gmmBoxes = gmm_detector(videoName, usableFrames, gtAllBoxes);

fprintf('Running tiny YOLO\n');
yoloBoxes = yolo_detector(videoName, usableFrames, gtAllBoxes);

fprintf('Running csp YOLO\n');
yolov2CarBoxes = yolo_csp_detector(videoName, usableFrames, gtAllBoxes);

% IOU threshold for metric calculation
iouThreshold = 0.5;

% Evaluating detectors
thresholdMetrics = evaluate_metrics(thresholdBoxes, gtAllBoxes, iouThreshold);
gmmMetrics = evaluate_metrics(gmmBoxes, gtAllBoxes, iouThreshold);
yoloMetrics = evaluate_metrics(yoloBoxes, gtAllBoxes, iouThreshold);
yolov2CarMetrics = evaluate_metrics(yolov2CarBoxes, gtAllBoxes, iouThreshold);

% Show results
display_evaluations(thresholdMetrics,gmmMetrics, yoloMetrics, yolov2CarMetrics);

% Thresholding
function allBoxes = threshold_detector(videoFile, totalFrames, groundTruth)
    % Using the first frame as static background
    reader = VideoReader(videoFile);
    refFrame = read(reader, 1);
    % converting it to gray scale
    refGray = rgb2gray(refFrame);
    allBoxes = cell(totalFrames, 1);

    for idx = 1:totalFrames
        currentFrame = read(reader, idx);
        grayFrame = rgb2gray(currentFrame);

        % Difference image (simple background subtraction)
        diffImg = abs(double(grayFrame) - double(refGray));

        % Binary threshold
        binaryMask = imbinarize(uint8(diffImg), 0.1);
        % Remove small items
        binaryMask = bwareaopen(binaryMask, 50);

        % Get bounding boxes
        regions = regionprops(binaryMask, 'BoundingBox');
        boxes = reshape([regions.BoundingBox], 4, []).';

        allBoxes{idx} = boxes;

        % Visualization
        subplot(2,3,1);
        imshow(currentFrame);
        title(sprintf('Original Frame %d', idx));
        subplot(2,3,2);
        imshow(grayFrame);
        title('Grayscale');
        subplot(2,3,3);
        imshow(uint8(diffImg));
        title('Absolute Difference');
        subplot(2,3,4);
        imshow(binaryMask);
        title('Binary Mask');

        overlayFrame = currentFrame;
        if ~isempty(boxes)
            overlayFrame = insertShape(overlayFrame, 'Rectangle', boxes, 'Color', 'red', 'LineWidth', 2);
        end
        if ~isempty(groundTruth{idx})
            overlayFrame = insertShape(overlayFrame, 'Rectangle', groundTruth{idx}, 'Color', 'green', 'LineWidth', 3);
        end

        subplot(2,3,5);
        imshow(overlayFrame);
        title('Detections (Red), Ground Truth (Green)');
        pause(0.1);
    end
end

% GMM Detector
function allBoxes = gmm_detector(videoPath, frameCap, gt)
    reader = VideoReader(videoPath);
    fgDetector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 5);
    allBoxes = cell(frameCap, 1);
    
    for frameNum = 1:frameCap
        current = read(reader, frameNum);
        gray = rgb2gray(current);
        mask = fgDetector.step(gray);
        % Remove small items
        mask = bwareaopen(mask, 60);
        
        stats = regionprops(mask, 'BoundingBox');
        boxes = reshape([stats.BoundingBox], 4, []).';
        allBoxes{frameNum} = boxes;
        % Visualize 
        subplot(1, 2, 1);
        imshow(mask);
        title(sprintf('GMM Foreground Mask - Frame %d', frameNum))
        % adding mask to show the detected cars
        if ~isempty(boxes)
            current = insertShape(current, 'Rectangle', boxes, 'Color', 'red', 'LineWidth', 2);
        end
        if ~isempty(gt{frameNum})
            current = insertShape(current, 'Rectangle', gt{frameNum}, 'Color', 'green', 'LineWidth', 3);
        end
        subplot(1,2,2); imshow(current);
        title(sprintf('GMM Detector Result - Frame %d', frameNum));
        pause(0.1);
    end
end

% tiny YOLO
function allBoxes = yolo_detector(videoPath, maxFrames, gt)

    reader = VideoReader(videoPath);
    detector = yolov4ObjectDetector("tiny-yolov4-coco");

    allBoxes = cell(maxFrames, 1);
    confidenceThreshold = 0.3;

    for frameIdx = 1:maxFrames
        frame = read(reader, frameIdx);
        [bboxes, scores, labels] = detect(detector, frame);

        % Filter by class and confidence
        carMask = strcmp(cellstr(labels), 'car') & scores > confidenceThreshold;
        carBoxes = bboxes(carMask, :);

        allBoxes{frameIdx} = carBoxes;

        if ~isempty(carBoxes)
            frame = insertShape(frame, 'Rectangle', carBoxes, 'Color', 'red', 'LineWidth', 2);
        end
        if ~isempty(gt{frameIdx})
            frame = insertShape(frame, 'Rectangle', gt{frameIdx}, 'Color', 'green', 'LineWidth', 3);
        end
        subplot(1, 2, 2); imshow(frame);
        title(sprintf('YOLO tiny-yolov4-coco detector - Frame: %d', frameIdx));
        pause(0.05);
    end
end

% csp YOLO
function allBoxes = yolo_csp_detector(vidFile, totalFrames, gtLabels)

    reader = VideoReader(vidFile);
    detector = yolov4ObjectDetector("csp-darknet53-coco");

    allBoxes = cell(totalFrames, 1);
    minConfidence = 0.3;

    for i = 1:totalFrames
        frame = read(reader, i);
        [bboxes, scores, labels] = detect(detector, frame);

        carBoxes = bboxes(strcmp(cellstr(labels), 'car') & scores > minConfidence, :);

        allBoxes{i} = carBoxes;

        if ~isempty(carBoxes)
            frame = insertShape(frame, 'Rectangle', carBoxes, 'Color', 'red', 'LineWidth', 2);
        end
        if ~isempty(gtLabels{i})
            frame = insertShape(frame, 'Rectangle', gtLabels{i}, 'Color', 'green', 'LineWidth', 3);
        end

        subplot(1, 2, 2); imshow(frame);
        title(sprintf('YOLO csp-darknet53-coco detector - Frame %d', i));
        pause(0.01);
    end
end

% Metrics Evaluation
function metrics = evaluate_metrics(predBoxes, gtBoxes, iouThreshold)
    numFrames = length(gtBoxes);
    precisions = zeros(numFrames,1);
    recalls = zeros(numFrames,1);
    for k = 1:numFrames
        gt = gtBoxes{k};
        pred = predBoxes{k};
        [p, r] = bboxPrecisionRecall(pred, gt, iouThreshold);
        precisions(k) = p;
        recalls(k) = r;
    end
    precision = mean(precisions);
    recall = mean(recalls);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
    metrics = struct('Precision', precision, 'Recall', recall, 'F1Score', f1);
end


function display_evaluations(thresholdMetrics,gmmMetrics, yoloMetrics, yolov2CarMetrics)
    disp('Thresholding:');
    disp(thresholdMetrics);
    disp('GMM:');
    disp(gmmMetrics);
    disp('YOLO tiny-yolov4-coco detector:');
    disp(yoloMetrics);
    disp('YOLO csp-darknet53-coco detector:');
    disp(yolov2CarMetrics);
    
    metricsNames = {'Threshold', 'GMM', 'YOLO tiny-yolov4-coco', 'YOLO csp-darknet53-coco'};
    precisions = [thresholdMetrics.Precision, gmmMetrics.Precision, yoloMetrics.Precision, yolov2CarMetrics.Precision];
    recalls = [thresholdMetrics.Recall, gmmMetrics.Recall, yoloMetrics.Recall, yolov2CarMetrics.Recall];
    f1Scores = [thresholdMetrics.F1Score, gmmMetrics.F1Score, yoloMetrics.F1Score, yolov2CarMetrics.F1Score];
    figure('Position', [100, 100, 1200, 700]);
    barData = [precisions; recalls; f1Scores]';
    barHandle = bar(barData, 'grouped');
    set(barHandle, 'FaceColor', 'flat');
    
    barHandle(1).FaceColor = [0.2 0.6 1];
    barHandle(2).FaceColor = [1 0.6 0];
    barHandle(3).FaceColor = [0.8 0 0]; 
    
    set(gca, 'XTickLabel', metricsNames);
    legend({'Precision', 'Recall', 'F1 Score'}, 'Location', 'Best');
    xlabel('Detector');
    ylabel('Metric Value');
    title('Evaluation Metrics for Different Detectors');
    grid on;
end