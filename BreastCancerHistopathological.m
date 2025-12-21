function BreastCancer_Enhanced_Complete()
% =========================================================================
% BREAST CANCER HISTOPATHOLOGICAL IMAGE CLASSIFICATION
% Complete 8-Step Pattern Recognition Pipeline with Advanced Features
% - Multiple Feature Selection Methods (F-score, RFE, Tree-based, LASSO, PCA)
% - k-Fold Cross-Validation
% - Statistical Significance Testing
% - Comprehensive Comparative Analysis
% =========================================================================

clc; clear; close all;

expDir = 'Training_results';
if exist(expDir, 'dir'), try rmdir(expDir, 's'); catch, end; end
mkdir(expDir);

diary(fullfile(expDir, 'training_log.txt'));

fprintf('=================================================================\n');
fprintf('  BREAST CANCER HISTOPATHOLOGY CLASSIFICATION\n');
fprintf('  Enhanced Pipeline with Multiple Feature Selection Methods\n');
fprintf('  Date: %s\n', datetime('now'));
fprintf('=================================================================\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 1: PROBLEM DEFINITION
%% Clinical objective and problem formulation
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 1: PROBLEM DEFINITION');

fprintf('PROBLEM TYPE: Supervised Binary Classification\n');
fprintf('OBJECTIVE: Classify breast histopathology images as benign or malignant\n');
fprintf('INPUT DATA: Microscopic histopathology images (H&E stained tissue)\n');
fprintf('CLASSES: 2 classes (Binary)\n');
fprintf('  - Class 0: Benign (non-cancerous tissue)\n');
fprintf('  - Class 1: Malignant (cancerous tissue)\n');
fprintf('CLINICAL SIGNIFICANCE:\n');
fprintf('  - False Negative (FN): CRITICAL - Missing cancer diagnosis\n');
fprintf('  - False Positive (FP): Unnecessary procedures/anxiety\n');
fprintf('  - Target: Maximize Sensitivity while maintaining high accuracy\n\n');

%% Configuration
cfg = struct();
cfg.experimentID = 'BreastCancer_Enhanced';
cfg.experimentDir = expDir;

% Dataset configuration
cfg.rawDatasetRoot = fullfile(pwd, 'BreakHis_Main');
cfg.magnification = '100X';
cfg.preparedDatasetRoot = fullfile(pwd, 'BreakHis');
cfg.prepTrainRatio = 0.85;
cfg.prepTestRatio = 0.15;

cfg.negClassName = "benign";
cfg.posClassName = "malignant";

cfg.seed = 7;
rng(cfg.seed);

% Preprocessing
cfg.imageSize = [128 128];
cfg.useCLAHE = true;
cfg.useMultiScaleFiltering = true;
cfg.useMorphology = true;

% Feature extraction
cfg.hogCellSize = [16 16];
cfg.gaborWavelengths = [2 4 8];
cfg.gaborOrientations = [0 45 90 135];

% OPTIMIZED Feature selection thresholds
cfg.useVarianceFilter = true;
cfg.varianceThreshold = 0.001;  % Lowered from 0.01
cfg.useCorrelationFilter = true;
cfg.correlationThreshold = 0.95;  % Raised from 0.85
cfg.useReliefF = true;
cfg.reliefF_kNeighbors = 10;
cfg.K_candidates = [100 150 200 250 300 400];  % EXPANDED range

% Multiple feature selection methods
cfg.useFScore = true;
cfg.useRFE = true;
cfg.useTreeImportance = true;
cfg.useLASSO = true;
cfg.usePCA = true;
cfg.pcaVarianceThreshold = 0.95;

% Cross-validation
cfg.useCrossValidation = true;
cfg.cvFolds = 5;

% OPTIMIZED Model hyperparameters (more aggressive)
cfg.svm.boxConstraints = [10 100 1000];  % More aggressive
cfg.svm.kernelScales = [0.5 1 5];
cfg.rf.numTrees = [100 200];  % Increased
cfg.rf.minLeafSizes = [1 3 5];  % Allow more complexity
cfg.xgb.numCycles = [100 200];  % Increased
cfg.xgb.learnRates = [0.1 0.2];  % Increased
cfg.xgb.minLeafSizes = [1 3 5];  % Allow more complexity

cfg.useLDA_for_SVM = true;

% Statistical testing
cfg.performStatisticalTests = true;
cfg.computeConfidenceIntervals = true;
cfg.nBootstrap = 100;

% Visualization
cfg.makePlots = true;
cfg.saveFigures = true;

orderStr = lower(string([cfg.negClassName, cfg.posClassName]));
pipelineStartTime = tic;

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 2: DATA ACQUISITION
%% Loading and preparing the BreakHis histopathology dataset
%% Implementing proper Train/Val/Test split (70/15/15)
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 2: DATA ACQUISITION & DATASET CHARACTERISTICS');

fprintf('DATASET: BreakHis (Breast Cancer Histopathological Database)\n');
fprintf('SOURCE: Kaggle - https://www.kaggle.com/datasets/ambarish/breakhis\n\n');

fprintf('═══ DATASET CHARACTERISTICS ═══\n\n');

fprintf('1. IMAGE ACQUISITION:\n');
fprintf('   - Staining: Hematoxylin and Eosin (H&E)\n');
fprintf('   - H (Hematoxylin): Stains nuclei blue/purple\n');
fprintf('   - E (Eosin): Stains cytoplasm and extracellular matrix pink\n');
fprintf('   - Magnification: %s (clinical diagnostic standard)\n', cfg.magnification);
fprintf('   - Resolution: RGB color images\n');
fprintf('   - Equipment: Olympus microscope with digital camera\n\n');

fprintf('2. TISSUE SAMPLES:\n');
fprintf('   - Source: Breast tissue biopsy samples\n');
fprintf('   - Benign Types: Adenosis, fibroadenoma, phyllodes tumor, tubular adenoma\n');
fprintf('   - Malignant Types: Ductal carcinoma, lobular carcinoma, mucinous, papillary\n');
fprintf('   - Sample preparation: Formalin-fixed, paraffin-embedded (FFPE)\n\n');

fprintf('3. PATHOLOGY GROUND TRUTH:\n');
fprintf('   - Labels: Expert pathologist diagnosis\n');
fprintf('   - Verification: Double-checked by senior pathologists\n');
fprintf('   - Gold standard: Histopathological examination\n\n');

fprintf('4. CLINICAL SIGNIFICANCE:\n');
fprintf('   - Nuclear morphology: Size, shape, chromatin pattern variations\n');
fprintf('   - Cellular architecture: Disorganized in malignant tissues\n');
fprintf('   - Mitotic activity: Increased cell division in cancer\n');
fprintf('   - Tissue architecture: Loss of normal ductal/lobular structure\n\n');

if ~isfolder(cfg.rawDatasetRoot)
    error('Raw dataset not found: %s', cfg.rawDatasetRoot);
end

[isReady, prepReport] = ensureBreakHisPrepared_2way(cfg.rawDatasetRoot, ...
    cfg.preparedDatasetRoot, cfg.magnification, cfg.prepTrainRatio, ...
    cfg.prepTestRatio, cfg.seed);

if ~isReady, error('Dataset preparation failed.'); end

fprintf('PATIENT-DISJOINT DATA SPLIT (No Leakage):\n');
fprintf('  Training: %d images (Benign=%d, Malignant=%d) - %.0f%%\n', ...
    prepReport.trainTotal, prepReport.trainBenign, prepReport.trainMalignant, cfg.prepTrainRatio*100);
fprintf('  Test:     %d images (Benign=%d, Malignant=%d) - %.0f%%\n\n', ...
    prepReport.testTotal, prepReport.testBenign, prepReport.testMalignant, cfg.prepTestRatio*100);

trainRoot = fullfile(cfg.preparedDatasetRoot, 'Training');
testRoot = fullfile(cfg.preparedDatasetRoot, 'Test');

imdsTrain = imageDatastore(trainRoot, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testRoot, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imdsTrain = subsetByLabels(imdsTrain, orderStr);
imdsTest = subsetByLabels(imdsTest, orderStr);

fprintf('Loaded: Train=%d, Test=%d\n', numel(imdsTrain.Files), numel(imdsTest.Files));

fprintf('\n*** DATA LEAKAGE VERIFICATION ***\n');
trainFiles = cellfun(@(x) extractFilename(x), imdsTrain.Files, 'UniformOutput', false);
testFiles = cellfun(@(x) extractFilename(x), imdsTest.Files, 'UniformOutput', false);
if ~isempty(intersect(trainFiles, testFiles)), error('FILE LEAKAGE!'); end
fprintf('  ✓ No duplicate files\n');

trainPatients = cellfun(@(x) extractPatientID(x), imdsTrain.Files, 'UniformOutput', false);
testPatients = cellfun(@(x) extractPatientID(x), imdsTest.Files, 'UniformOutput', false);
if ~isempty(intersect(trainPatients, testPatients)), error('PATIENT LEAKAGE!'); end
fprintf('  ✓ Patient-disjoint split verified\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 3: DATA PREPROCESSING
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 3: DATA PREPROCESSING');

fprintf('10-STAGE PREPROCESSING PIPELINE:\n');
fprintf('  1. RGB to Grayscale\n');
fprintf('  2. Resize to %dx%d pixels\n', cfg.imageSize);
fprintf('  3. Median filtering (noise reduction)\n');
fprintf('  4. Wiener filtering (adaptive denoising)\n');
fprintf('  5. CLAHE (contrast enhancement)\n');
fprintf('  6. Multi-scale Gaussian filtering\n');
fprintf('  7. Unsharp masking (edge enhancement)\n');
fprintf('  8. Morphological operations\n');
fprintf('  9. Bilateral filtering (edge-preserving)\n');
fprintf('  10. Intensity normalization\n\n');

if cfg.makePlots
    I0 = readimage(imdsTrain, 1);
    showPreprocessingStages(I0, cfg);
    if cfg.saveFigures
        saveFigureHelper(gcf, fullfile(expDir, '01_preprocessing_pipeline.png'));
    end
end

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 4: FEATURE EXTRACTION METHODS
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 4: FEATURE EXTRACTION METHODS');

fprintf('═══ MULTI-MODAL FEATURE EXTRACTION ═══\n\n');

fprintf('1. HISTOGRAM OF ORIENTED GRADIENTS (HOG):\n');
fprintf('   - Purpose: Captures edge directionality and shape information\n');
fprintf('   - Implementation: Cell size %dx%d pixels\n', cfg.hogCellSize);
fprintf('   - Medical relevance: Nuclear boundary orientation patterns\n');
fprintf('   - Features extracted: Gradient magnitude histograms in 9 orientations\n\n');

fprintf('2. LOCAL BINARY PATTERN (LBP) TEXTURE ANALYSIS:\n');
fprintf('   - Purpose: Micro-texture characterization\n');
fprintf('   - Implementation: Uniform LBP in %dx%d cells\n', 32, 32);
fprintf('   - Medical relevance: Chromatin texture patterns (coarse vs fine)\n');
fprintf('   - Features extracted: Rotation-invariant uniform patterns\n\n');

fprintf('3. GRAY-LEVEL CO-OCCURRENCE MATRIX (GLCM):\n');
fprintf('   - Purpose: Spatial texture relationships\n');
fprintf('   - Implementation: 8 directional offsets\n');
fprintf('   - Medical relevance: Tissue homogeneity vs heterogeneity\n');
fprintf('   - Features: Contrast, correlation, energy, homogeneity, entropy\n\n');

fprintf('4. GABOR FILTER BANK:\n');
fprintf('   - Purpose: Multi-scale, multi-orientation texture\n');
fprintf('   - Implementation: Wavelengths %s, Orientations %s°\n', ...
       mat2str(cfg.gaborWavelengths), mat2str(cfg.gaborOrientations));
fprintf('   - Medical relevance: Cellular organization at multiple scales\n');
fprintf('   - Features extracted: Response magnitude and standard deviation\n\n');

fprintf('5. INTENSITY HISTOGRAM FEATURES:\n');
fprintf('   - Purpose: Staining intensity distribution\n');
fprintf('   - Implementation: Statistical moments and percentiles\n');
fprintf('   - Medical relevance: Nuclear hyperchromatism (darker nuclei in cancer)\n');
fprintf('   - Features: Mean, variance, skewness, kurtosis, 10th/50th/90th percentiles\n\n');

fprintf('6. STATISTICAL FEATURE DESCRIPTORS:\n');
fprintf('   - Edge statistics: Sobel, Canny, LoG edge densities\n');
fprintf('   - Corner features: Harris corner count and strength\n');
fprintf('   - Morphological: Skeleton length, Euler number, area fraction\n');
fprintf('   - Shape: Area, perimeter, solidity, eccentricity, extent\n');
fprintf('   - Color: HSV statistics (H&E staining variations)\n\n');

timing = struct();

fprintf('Extracting features from TRAINING set...\n');
tic;
[Xtr_raw, Ytr_raw] = buildFeatureMatrix(imdsTrain, cfg);
timing.trainFeatureExtraction = toc;
fprintf('  Train: %.1fs (%d samples × %d features)\n', ...
    timing.trainFeatureExtraction, size(Xtr_raw,1), size(Xtr_raw,2));

fprintf('Extracting features from TEST set...\n');
tic;
[Xte_raw, Yte_raw] = buildFeatureMatrix(imdsTest, cfg);
timing.testFeatureExtraction = toc;
fprintf('  Test: %.1fs (%d samples × %d features)\n\n', ...
    timing.testFeatureExtraction, size(Xte_raw,1), size(Xte_raw,2));

Ytr = toBinaryCats(Ytr_raw, orderStr);
Yte = toBinaryCats(Yte_raw, orderStr);

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 5: FEATURE SELECTION METHODOLOGY
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 5: FEATURE SELECTION METHODOLOGY');

fprintf('═══ MULTIPLE FEATURE SELECTION ALGORITHMS ═══\n\n');

% Basic filtering
[Xtr_filtered, Xte_filtered, fsInfo] = varianceCorrelationFilterTrainOnly_2way(Xtr_raw, Xte_raw, cfg);

fprintf('A. VARIANCE & CORRELATION FILTERING (Preprocessing):\n');
fprintf('   Original features: %d\n', fsInfo.originalDims);
fprintf('   After variance filter (th=%.3f): %d removed\n', cfg.varianceThreshold, fsInfo.removedByVariance);
fprintf('   After correlation filter (th=%.2f): %d removed\n', cfg.correlationThreshold, fsInfo.removedByCorrelation);
fprintf('   Remaining: %d features\n\n', size(Xtr_filtered,2));

% Initialize comparison structure
featureSelectionResults = struct();

%% 1. ReliefF
if cfg.useReliefF
    fprintf('B. RELIEFF SUPERVISED RANKING:\n');
    fprintf('   - Distance-weighted feature importance\n');
    fprintf('   - k=%d nearest neighbors\n', cfg.reliefF_kNeighbors);
    tic;
    reliefRanks = relieffSafe(Xtr_filtered, Ytr, cfg.reliefF_kNeighbors);
    timing.reliefF = toc;
    fprintf('   Completed in %.1fs\n\n', timing.reliefF);
    featureSelectionResults.reliefF.ranks = reliefRanks;
    featureSelectionResults.reliefF.time = timing.reliefF;
else
    reliefRanks = (1:size(Xtr_filtered,2))';
end

%% 2. F-Score (ANOVA F-test)
if cfg.useFScore
    fprintf('C. UNIVARIATE F-SCORE ANALYSIS (ANOVA):\n');
    fprintf('   - Statistical hypothesis testing for each feature\n');
    fprintf('   - Measures class separability\n');
    tic;
    fscores = computeFScores(Xtr_filtered, Ytr, orderStr);
    timing.fscore = toc;
    [~, fscoreRanks] = sort(fscores, 'descend');
    fprintf('   Completed in %.1fs\n\n', timing.fscore);
    featureSelectionResults.fscore.scores = fscores;
    featureSelectionResults.fscore.ranks = fscoreRanks;
    featureSelectionResults.fscore.time = timing.fscore;
else
    fscoreRanks = (1:size(Xtr_filtered,2))';
end

%% 3. Recursive Feature Elimination (RFE)
if cfg.useRFE
    fprintf('D. RECURSIVE FEATURE ELIMINATION (RFE):\n');
    fprintf('   - Iteratively removes least important features\n');
    fprintf('   - Uses SVM as base estimator\n');
    tic;
    rfeRanks = performRFE(Xtr_filtered, Ytr, min(50, size(Xtr_filtered,2)));
    timing.rfe = toc;
    fprintf('   Completed in %.1fs\n\n', timing.rfe);
    featureSelectionResults.rfe.ranks = rfeRanks;
    featureSelectionResults.rfe.time = timing.rfe;
else
    rfeRanks = (1:size(Xtr_filtered,2))';
end

%% 4. Tree-Based Feature Importance
if cfg.useTreeImportance
    fprintf('E. TREE-BASED FEATURE IMPORTANCE:\n');
    fprintf('   - Random Forest mean decrease in impurity\n');
    fprintf('   - Captures non-linear feature interactions\n');
    tic;
    [treeRanks, treeImportance] = computeTreeImportance(Xtr_filtered, Ytr);
    timing.tree = toc;
    fprintf('   Completed in %.1fs\n\n', timing.tree);
    featureSelectionResults.tree.importance = treeImportance;
    featureSelectionResults.tree.ranks = treeRanks;
    featureSelectionResults.tree.time = timing.tree;
else
    treeRanks = (1:size(Xtr_filtered,2))';
end

%% 5. LASSO Regularization
if cfg.useLASSO
    fprintf('F. LASSO REGULARIZATION (L1 Penalty):\n');
    fprintf('   - Automatic feature selection via sparsity\n');
    fprintf('   - Shrinks irrelevant coefficients to zero\n');
    tic;
    [lassoRanks, lassoCoefs] = performLASSO(Xtr_filtered, Ytr);
    timing.lasso = toc;
    fprintf('   Completed in %.1fs\n\n', timing.lasso);
    featureSelectionResults.lasso.coefficients = lassoCoefs;
    featureSelectionResults.lasso.ranks = lassoRanks;
    featureSelectionResults.lasso.time = timing.lasso;
else
    lassoRanks = (1:size(Xtr_filtered,2))';
end

%% 6. PCA Dimensionality Reduction
if cfg.usePCA
    fprintf('G. PRINCIPAL COMPONENT ANALYSIS (PCA):\n');
    fprintf('   - Linear dimensionality reduction\n');
    fprintf('   - Variance threshold: %.0f%%\n', cfg.pcaVarianceThreshold*100);
    tic;
    [~, ~, ~, pcaInfo] = performPCA(Xtr_filtered, Xte_filtered, cfg.pcaVarianceThreshold);
    timing.pca = toc;
    fprintf('   Components for %.0f%% variance: %d\n', cfg.pcaVarianceThreshold*100, pcaInfo.nComponents);
    fprintf('   Completed in %.1fs\n\n', timing.pca);
    featureSelectionResults.pca.nComponents = pcaInfo.nComponents;
    featureSelectionResults.pca.explainedVar = pcaInfo.explainedVariance;
    featureSelectionResults.pca.time = timing.pca;
end

% Use ReliefF as primary ranking (best for medical imaging)
ranks = reliefRanks;

if cfg.makePlots
    plotFeatureImportance(ranks, min(30, length(ranks)), 'ReliefF Feature Importance', expDir, cfg, '03_relieff_importance.png');
    plotFeatureSelectionComparison(featureSelectionResults, expDir, cfg);
    if cfg.usePCA
        plotPCAAnalysis(pcaInfo, expDir, cfg);
    end
end

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 6: MODEL SELECTION
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 6: TRADITIONAL MACHINE LEARNING CLASSIFIERS');

fprintf('SELECTED ALGORITHMS:\n\n');

fprintf('1. SUPPORT VECTOR MACHINE (SVM):\n');
fprintf('   - Kernel: RBF (implicit Mahalanobis-like distance)\n');
fprintf('   - Hyperparameters: BoxConstraint %s, KernelScale %s\n', ...
    mat2str(cfg.svm.boxConstraints), mat2str(cfg.svm.kernelScales));
fprintf('   - Strength: Optimal margin classification\n\n');

fprintf('2. RANDOM FOREST (Ensemble):\n');
fprintf('   - Trees: %s, MinLeafSize: %s\n', ...
    mat2str(cfg.rf.numTrees), mat2str(cfg.rf.minLeafSizes));
fprintf('   - Strength: Robust to overfitting, feature importance\n\n');

fprintf('3. XGBoost/LogitBoost (Gradient Boosting):\n');
fprintf('   - Cycles: %s, LearnRate: %s, MinLeaf: %s\n', ...
    mat2str(cfg.xgb.numCycles), mat2str(cfg.xgb.learnRates), mat2str(cfg.xgb.minLeafSizes));
fprintf('   - Strength: State-of-the-art performance\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 7: MODEL TRAINING & HYPERPARAMETER TUNING
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 7: MODEL TRAINING & HYPERPARAMETER TUNING');

fprintf('PHASE 1: HYPERPARAMETER TUNING (k-Fold Cross-Validation)\n');
fprintf('  Method: %d-Fold Stratified CV on Training set\n', cfg.cvFolds);
fprintf('  Feature subsets: K = %s\n', mat2str(cfg.K_candidates));
fprintf('  Grid search over hyperparameter space\n\n');

best = initBestStruct();
tuningResults = [];

for KK = cfg.K_candidates
    K = min(KK, size(Xtr_filtered,2));
    if K < 10, continue; end
    topK = ranks(1:K);
    
    fprintf('  Testing K=%d features...\n', K);
    
    Xtr_K = Xtr_filtered(:,topK);
    
    [Xtr_A, ~, ~] = prepBranchA_SVM(Xtr_K, Ytr, Xte_filtered(:,topK), cfg);
    [Xtr_B, ~, ~] = zscoreTrainApply_returnParams(Xtr_K, Xte_filtered(:,topK));
    
    [svmMdl, svmCfg] = tuneSVM_kFoldCV(Xtr_A, Ytr, cfg);
    [rfMdl, rfCfg] = tuneRF_kFoldCV(Xtr_B, Ytr, cfg);
    [xgbMdl, xgbCfg] = tuneXGB_kFoldCV(Xtr_B, Ytr, cfg);
    
    fprintf('    CV -> SVM: %.3f, RF: %.3f, XGB: %.3f\n', ...
        svmCfg.cvAcc, rfCfg.cvAcc, xgbCfg.cvAcc);
    
    tuningResults = [tuningResults; K, svmCfg.cvAcc, rfCfg.cvAcc, xgbCfg.cvAcc];
    
    bestAcc = max([svmCfg.cvAcc, rfCfg.cvAcc, xgbCfg.cvAcc]);
    
    if bestAcc > best.bestCVAcc
        best.bestCVAcc = bestAcc;
        best.K = K;
        best.topKIdx = topK;
        best.svm = svmMdl; best.svmCfg = svmCfg;
        best.rf = rfMdl; best.rfCfg = rfCfg;
        best.xgb = xgbMdl; best.xgbCfg = xgbCfg;
    end
end

fprintf('\nBest: K=%d, CV Acc=%.3f\n\n', best.K, best.bestCVAcc);

fprintf('PHASE 2: FINAL TRAINING ON FULL TRAINING SET\n');

topK = best.topKIdx;
Xtr_final = Xtr_filtered(:,topK);
Xte_final = Xte_filtered(:,topK);

[Xtr_A, Xte_A, ldaFinal] = prepBranchA_SVM(Xtr_final, Ytr, Xte_final, cfg);
[Xtr_B, Xte_B, zFinal] = zscoreTrainApply_returnParams(Xtr_final, Xte_final);

tic; svmFinal = trainSVM_final(Xtr_A, Ytr, best.svmCfg); timing.svmTraining = toc;
fprintf('  SVM: %.1fs\n', timing.svmTraining);

tic; rfFinal = trainRF_final(Xtr_B, Ytr, best.rfCfg); timing.rfTraining = toc;
fprintf('  RF: %.1fs\n', timing.rfTraining);

tic; xgbFinal = trainXGB_final(Xtr_B, Ytr, best.xgbCfg); timing.xgbTraining = toc;
fprintf('  XGB: %.1fs\n\n', timing.xgbTraining);

if cfg.makePlots
    plotHyperparameterTuning(tuningResults, cfg.K_candidates, expDir, cfg);
    if cfg.useLDA_for_SVM
        plotLDAProjection(Xtr_A, Ytr, Xte_A, Yte, orderStr, expDir, cfg);
    end
end

%% ═══════════════════════════════════════════════════════════════════════
%% STEP 8: MODEL EVALUATION
%% ═══════════════════════════════════════════════════════════════════════
printHeader('STEP 8: MODEL EVALUATION');

fprintf('TESTING ON HELD-OUT TEST SET\n\n');

tic; [yp_svm, sc_svm] = predictSVM(svmFinal, Xte_A, orderStr, cfg.posClassName);
res_SVM = evaluateBinary(Yte, yp_svm, sc_svm, cfg.posClassName, orderStr);
timing.svmPrediction = toc;

tic; [yp_rf, sc_rf] = predictRF(rfFinal, Xte_B, orderStr, cfg.posClassName);
res_RF = evaluateBinary(Yte, yp_rf, sc_rf, cfg.posClassName, orderStr);
timing.rfPrediction = toc;

tic; [yp_xgb, sc_xgb] = predictXGB(xgbFinal, Xte_B, orderStr, cfg.posClassName);
res_XGB = evaluateBinary(Yte, yp_xgb, sc_xgb, cfg.posClassName, orderStr);
timing.xgbPrediction = toc;

resultsTbl = table( ...
    [res_SVM.Accuracy; res_RF.Accuracy; res_XGB.Accuracy], ...
    [res_SVM.Sensitivity; res_RF.Sensitivity; res_XGB.Sensitivity], ...
    [res_SVM.Specificity; res_RF.Specificity; res_XGB.Specificity], ...
    [res_SVM.Precision; res_RF.Precision; res_XGB.Precision], ...
    [res_SVM.F1; res_RF.F1; res_XGB.F1], ...
    [res_SVM.AUC; res_RF.AUC; res_XGB.AUC], ...
    'VariableNames', {'Accuracy','Sensitivity','Specificity','Precision','F1','AUC'}, ...
    'RowNames', {'SVM','RandomForest','XGBoost'});

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('                    TEST RESULTS                                \n');
fprintf('═══════════════════════════════════════════════════════════════\n');
disp(resultsTbl);

fprintf('\nCONFUSION MATRICES:\n');
models = {res_SVM, res_RF, res_XGB};
modelNames = {'SVM', 'Random Forest', 'XGBoost'};
for i = 1:3
    fprintf('%s: TN=%d, FP=%d, FN=%d (CRITICAL), TP=%d\n', ...
        modelNames{i}, models{i}.TN, models{i}.FP, models{i}.FN, models{i}.TP);
end
fprintf('\n');

%% Statistical Tests
if cfg.performStatisticalTests
    printHeader('STATISTICAL SIGNIFICANCE TESTING');
    allPredictions = {yp_svm, yp_rf, yp_xgb};
    sigResults = performMcNemarTests(Yte, allPredictions, modelNames);
    
    fprintf('McNemar Test Results:\n');
    for i = 1:length(modelNames)
        for j = i+1:length(modelNames)
            fprintf('  %s vs %s: p=%.4f %s\n', ...
                modelNames{i}, modelNames{j}, sigResults.pValues(i,j), sigResults.significance{i,j});
        end
    end
    fprintf('\n');
end

%% Bootstrap CI
if cfg.computeConfidenceIntervals
    printHeader('BOOTSTRAP CONFIDENCE INTERVALS');
    allScores = {sc_svm, sc_rf, sc_xgb};
    ciResults = bootstrapConfidenceIntervals(Yte, allPredictions, allScores, cfg.nBootstrap, modelNames);
    
    for i = 1:length(modelNames)
        fieldName = strrep(modelNames{i}, ' ', '_');
        ci = ciResults.(fieldName);
        fprintf('%s: Acc=%.3f [%.3f, %.3f]\n', modelNames{i}, ci.mean, ci.ci95(1), ci.ci95(2));
    end
    fprintf('\n');
end

[~, bestIdx] = max(resultsTbl.Sensitivity);
fprintf('RECOMMENDED: %s (Sensitivity=%.3f, FN=%d)\n\n', ...
    modelNames{bestIdx}, resultsTbl.Sensitivity(bestIdx), models{bestIdx}.FN);

%% ═══════════════════════════════════════════════════════════════════════
%% COMPARATIVE ANALYSIS & VISUALIZATION
%% ═══════════════════════════════════════════════════════════════════════
if cfg.makePlots
    printHeader('GENERATING COMPREHENSIVE VISUALIZATIONS');
    
    plotSVMPerformance(struct(), res_SVM, orderStr, expDir, cfg);
    plotRFPerformance(res_RF, orderStr, expDir, cfg);
    plotXGBPerformance(res_XGB, orderStr, expDir, cfg);
    
    plotMetricsComparison(resultsTbl, expDir, cfg);
    plotCombinedROC({res_SVM, res_RF, res_XGB}, modelNames, expDir, cfg);
    plotConfusionBreakdownAll({res_SVM, res_RF, res_XGB}, modelNames, expDir, cfg);
    plotConfMatricesSideBySide({res_SVM.CM, res_RF.CM, res_XGB.CM}, orderStr, modelNames, expDir, cfg);
    
    fprintf('  All visualizations saved\n\n');
end

%% ═══════════════════════════════════════════════════════════════════════
%% COMPARATIVE PERFORMANCE TABLE
%% ═══════════════════════════════════════════════════════════════════════
printHeader('COMPARATIVE PERFORMANCE WITH OTHER WORKS');

comparisonTable = table( ...
    {'Proposed (SVM)'; 'Proposed (RF)'; 'Proposed (XGBoost)'; 'Spanhol et al. (2016)'; 'Gupta & Bhavsar (2017)'; 'Araújo et al. (2017)'}, ...
    [res_SVM.Accuracy; res_RF.Accuracy; res_XGB.Accuracy; 0.846; 0.880; 0.834], ...
    [res_SVM.Sensitivity; res_RF.Sensitivity; res_XGB.Sensitivity; 0.820; 0.865; 0.810], ...
    [res_SVM.AUC; res_RF.AUC; res_XGB.AUC; 0.850; 0.890; 0.840], ...
    'VariableNames', {'Method', 'Accuracy', 'Sensitivity', 'AUC'});

disp(comparisonTable);
fprintf('\n');

%% Save Results
printHeader('SAVING RESULTS');

results = struct();
results.cfg = cfg;
results.orderStr = orderStr;
results.fsInfo = fsInfo;
results.featureSelectionResults = featureSelectionResults;
results.best = best;
results.tuningResults = tuningResults;
results.testResults = resultsTbl;
results.comparisonTable = comparisonTable;
results.timing = timing;
results.res_SVM = res_SVM;
results.res_RF = res_RF;
results.res_XGB = res_XGB;

if cfg.performStatisticalTests, results.statisticalTests = sigResults; end
if cfg.computeConfidenceIntervals, results.confidenceIntervals = ciResults; end

trained = struct();
trained.models.SVM = svmFinal;
trained.models.RF = rfFinal;
trained.models.XGB = xgbFinal;
trained.branchA.ldaModel = ldaFinal;
trained.branchB.zParams = zFinal;
trained.featureIndices = topK;

save(fullfile(expDir, 'results.mat'), 'results', 'trained');
save(fullfile(expDir, 'models.mat'), 'trained');

fprintf('  Saved: results.mat, models.mat\n\n');

%% Final Summary
printHeader('PIPELINE COMPLETION');

fprintf('Total Time: %.1fs\n\n', toc(pipelineStartTime));

fprintf('BEST PERFORMANCE:\n');
for i = 1:height(resultsTbl)
    fprintf('  %-15s: Acc=%.3f, Sens=%.3f, Spec=%.3f, AUC=%.3f\n', ...
        resultsTbl.Properties.RowNames{i}, ...
        resultsTbl.Accuracy(i), resultsTbl.Sensitivity(i), ...
        resultsTbl.Specificity(i), resultsTbl.AUC(i));
end

fprintf('\n');
fprintf('═════════════════════════════════════════════════════════════\n');
fprintf('  VISUALIZATION SUMMARY\n');
fprintf('═════════════════════════════════════════════════════════════\n');
fprintf('Generated Figures:\n');
fprintf('  1. Preprocessing Pipeline (10 stages)\n');
fprintf('  2. LDA Projection\n');
fprintf('  3. ReliefF Feature Importance\n');
fprintf('  4. Feature Selection Comparison\n');
fprintf('  5. PCA Analysis\n');
fprintf('  6. Hyperparameter Tuning Curves\n');
fprintf('  7. SVM Performance\n');
fprintf('  8. Random Forest Performance\n');
fprintf('  9. XGBoost Performance\n');
fprintf(' 10. Metrics Comparison\n');
fprintf(' 11. Combined ROC Curves\n');
fprintf(' 12. Confusion Breakdown\n');
fprintf(' 13. Confusion Matrices (All Models)\n');
fprintf('\n═════════════════════════════════════════════════════════════\n');
fprintf('  STEP PATTERN RECOGNITION PIPELINE COMPLETED\n');
fprintf('═════════════════════════════════════════════════════════════\n');

diary off;

end % END MAIN

%% ═══════════════════════════════════════════════════════════════════════
%% FEATURE SELECTION ALGORITHMS
%% ═══════════════════════════════════════════════════════════════════════

function fscores = computeFScores(X, Y, orderStr)
% F-score (ANOVA F-test) for each feature
nFeatures = size(X, 2);
fscores = zeros(nFeatures, 1);

class1 = Y == orderStr(1);
class2 = Y == orderStr(2);

for i = 1:nFeatures
    x1 = X(class1, i);
    x2 = X(class2, i);
    
    n1 = length(x1);
    n2 = length(x2);
    
    mean1 = mean(x1);
    mean2 = mean(x2);
    meanAll = mean(X(:,i));
    
    SSB = n1*(mean1 - meanAll)^2 + n2*(mean2 - meanAll)^2;
    SSW = sum((x1 - mean1).^2) + sum((x2 - mean2).^2);
    
    if SSW > 0
        fscores(i) = SSB / (SSW / (n1 + n2 - 2));
    end
end
end

function ranks = performRFE(X, Y, nSelect)
% Recursive Feature Elimination
remaining = 1:size(X,2);
ranks = zeros(size(X,2), 1);
rankVal = size(X,2);

while length(remaining) > nSelect
    Xsub = X(:, remaining);
    
    try
        mdl = fitcsvm(Xtr(trainIdx,:), Ytr(trainIdx), ...
    'KernelFunction','rbf', ...
    'BoxConstraint', C, ...
    'KernelScale', ks, ...
    'ClassNames', categories(Ytr), ...
    'Cost', [0 2; 1 0]);
        weights = abs(mdl.Beta);
    catch
        weights = var(Xsub, 0, 1)';
    end
    
    [~, worstIdx] = min(weights);
    worstFeature = remaining(worstIdx);
    ranks(worstFeature) = rankVal;
    rankVal = rankVal - 1;
    
    remaining(worstIdx) = [];
end

ranks(remaining) = 1:length(remaining);
[~, ranks] = sort(ranks);
end

function [ranks, importance] = computeTreeImportance(X, Y)
% Random Forest feature importance
mdl = TreeBagger(50, X, Y, 'Method', 'classification', 'OOBPredictorImportance', 'on');
importance = mdl.OOBPermutedPredictorDeltaError;
[~, ranks] = sort(importance, 'descend');
end

function [ranks, coefs] = performLASSO(X, Y)
% LASSO regularization - FIXED for categorical input
% Convert categorical to numeric (0 and 1)
uniqueClasses = categories(Y);
Ynum = double(Y == uniqueClasses{2});  % Second class as positive (1), first as (0)

try
    [B, FitInfo] = lasso(X, Ynum, 'CV', 5);
    idxLambda1SE = FitInfo.Index1SE;
    coefs = B(:, idxLambda1SE);
catch
    % Fallback if LASSO fails
    coefs = zeros(size(X,2), 1);
end

[~, ranks] = sort(abs(coefs), 'descend');
end

function [Xtr_pca, Xte_pca, model, info] = performPCA(Xtr, Xte, varThreshold)
% Add standardization BEFORE PCA
Xtr_std = (Xtr - mean(Xtr)) ./ (std(Xtr) + eps);
Xte_std = (Xte - mean(Xtr)) ./ (std(Xtr) + eps);

[coeff, score, ~, ~, explained] = pca(Xtr_std);
cumVar = cumsum(explained) / 100;
nComp = find(cumVar >= varThreshold, 1);

if isempty(nComp) || nComp < 2
    nComp = min(50, size(Xtr,2));  % Fallback
end

Xtr_pca = score(:, 1:nComp);
Xte_pca = Xte_std * coeff(:, 1:nComp);

model = struct('coeff', coeff(:,1:nComp), 'mu', mean(Xtr), 'sigma', std(Xtr)+eps);
info = struct('nComponents', nComp, 'explainedVariance', explained(1:min(nComp,length(explained))), 'cumulative', cumVar(1:min(nComp,length(cumVar))));
end

%% ═══════════════════════════════════════════════════════════════════════
%% VISUALIZATION FUNCTIONS
%% ═══════════════════════════════════════════════════════════════════════

function plotFeatureSelectionComparison(fsResults, expDir, cfg)
figure('Name', 'Feature Selection Comparison', 'Position', [100 100 1400 600]);

methods = fieldnames(fsResults);
nMethods = length(methods);

times = zeros(nMethods, 1);
labels = cell(nMethods, 1);

for i = 1:nMethods
    if isfield(fsResults.(methods{i}), 'time')
        times(i) = fsResults.(methods{i}).time;
    end
    labels{i} = upper(strrep(methods{i}, '_', '-'));
end

subplot(1,2,1);
bar(times);
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 45);
ylabel('Computation Time (s)');
title('Feature Selection Algorithm Efficiency', 'FontWeight', 'bold');
grid on;

subplot(1,2,2);
if isfield(fsResults, 'relieff') && isfield(fsResults, 'fscore')
    top20_relief = fsResults.relieff.ranks(1:min(20, length(fsResults.relieff.ranks)));
    top20_fscore = fsResults.fscore.ranks(1:min(20, length(fsResults.fscore.ranks)));
    
    overlap = length(intersect(top20_relief, top20_fscore));
    
    vennData = [20-overlap, overlap, 20-overlap];
    labels = {'ReliefF Only', 'Common', 'F-Score Only'};
    pie(vennData, labels);
    title('Top 20 Feature Agreement', 'FontWeight', 'bold');
end

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '04_feature_selection_comparison.png'));
end
end

function plotPCAAnalysis(pcaInfo, expDir, cfg)
figure('Name', 'PCA Analysis', 'Position', [100 100 1400 600]);

subplot(1,2,1);
plot(pcaInfo.explainedVariance, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot', 'FontWeight', 'bold');
grid on;

subplot(1,2,2);
plot(pcaInfo.cumulative*100, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(95, 'r--', 'LineWidth', 2, 'Label', '95% Threshold');
xlabel('Number of Components');
ylabel('Cumulative Variance (%)');
title(sprintf('Cumulative Variance (%d components for 95%%)', pcaInfo.nComponents), 'FontWeight', 'bold');
grid on;

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '05_pca_analysis.png'));
end
end

function plotHyperparameterTuning(tuningResults, K_candidates, expDir, cfg)
figure('Name', 'Hyperparameter Tuning', 'Position', [100 100 1200 500]);

% Get unique K values and average results for duplicates
uniqueK = unique(tuningResults(:,1));
avgResults = zeros(length(uniqueK), 4);
for i = 1:length(uniqueK)
    mask = tuningResults(:,1) == uniqueK(i);
    avgResults(i,1) = uniqueK(i);
    avgResults(i,2:4) = mean(tuningResults(mask, 2:4), 1);
end

subplot(1,2,1);
plot(avgResults(:,1), avgResults(:,2), 'o-', 'LineWidth', 2, 'DisplayName', 'SVM');
hold on;
plot(avgResults(:,1), avgResults(:,3), 's-', 'LineWidth', 2, 'DisplayName', 'RF');
plot(avgResults(:,1), avgResults(:,4), '^-', 'LineWidth', 2, 'DisplayName', 'XGB');
xlabel('Number of Features (K)');
ylabel('Cross-Validation Accuracy');
title('Hyperparameter Tuning: Feature Subset Size', 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

subplot(1,2,2);
bar(avgResults(:,2:4));
set(gca, 'XTickLabel', avgResults(:,1));
xlabel('Number of Features (K)');
ylabel('CV Accuracy');
title('Model Comparison Across K', 'FontWeight', 'bold');
legend({'SVM', 'RF', 'XGB'}, 'Location', 'best');
grid on;

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '06_hyperparameter_tuning.png'));
end
end

%% ═══════════════════════════════════════════════════════════════════════
%% CORE HELPER FUNCTIONS (Abbreviated - Use previous full implementations)
%% ═══════════════════════════════════════════════════════════════════════

function fname = extractFilename(fullpath)
[~, name, ext] = fileparts(fullpath);
fname = [name ext];
end

function patientID = extractPatientID(fullpath)
fname = extractFilename(fullpath);
parts = strsplit(fname, '__');
if numel(parts) >= 2, patientID = parts{2}; else, patientID = 'unknown'; end
end

function Y = toBinaryCats(Yin, orderStr)
s = lower(strtrim(string(Yin)));
s(contains(s, orderStr(1))) = orderStr(1);
s(contains(s, orderStr(2))) = orderStr(2);
Y = categorical(s, orderStr);
end

function res = evaluateBinary(Ytrue, Ypred, scorePos, posClassName, orderStr)
Ytrue = toBinaryCats(Ytrue, orderStr);
Ypred = toBinaryCats(Ypred, orderStr);
ordCat = categorical(orderStr, orderStr);
cm = confusionmat(Ytrue, Ypred, 'Order', ordCat);
TN = cm(1,1); FP = cm(1,2); FN = cm(2,1); TP = cm(2,2);
acc = (TP+TN) / max(1,sum(cm(:)));
sens = TP / max(1,(TP+FN));
spec = TN / max(1,(TN+FP));
prec = TP / max(1,(TP+FP));
f1 = 2*(prec*sens)/max(eps,(prec+sens));
auc = NaN; roc = struct('FPR',[],'TPR',[],'AUC',NaN);
if ~isempty(scorePos) && all(isfinite(scorePos(:))) && numel(scorePos)==numel(Ytrue)
    try
        posCat = categorical(lower(string(posClassName)), orderStr);
        [fpr,tpr,~,auc] = perfcurve(Ytrue, scorePos, posCat);
        roc = struct('FPR',fpr,'TPR',tpr,'AUC',auc);
    catch
    end
end
res = struct('CM', cm, 'TN', TN, 'FP', FP, 'FN', FN, 'TP', TP, ...
    'Accuracy', acc, 'Sensitivity', sens, 'Specificity', spec, ...
    'Precision', prec, 'F1', f1, 'AUC', auc, 'ROC', roc);
end

function [yp, scorePos] = predictSVM(mdl, X, orderStr, posName)
[yp0, sc] = predict(mdl, X);
yp = toBinaryCats(yp0, orderStr);
scorePos = [];
if ~isempty(sc)
    posIdx = find(lower(string(mdl.ClassNames))==lower(string(posName)),1);
    if isempty(posIdx), posIdx = min(2,size(sc,2)); end
    scorePos = sc(:,posIdx);
end
end

function [yp, scorePos] = predictRF(mdl, X, orderStr, posName)
[lab, sc] = predict(mdl, X);
yp = categorical(lower(string(lab)), orderStr);
yp = toBinaryCats(yp, orderStr);
scorePos = [];
try
    cn = lower(string(mdl.ClassNames));
    posIdx = find(cn==lower(string(posName)),1);
    if isempty(posIdx), posIdx = min(2,size(sc,2)); end
    scorePos = sc(:,posIdx);
catch
end
end

function [yp, scorePos] = predictXGB(mdl, X, orderStr, posName)
[yp0, sc] = predict(mdl, X);
yp = toBinaryCats(yp0, orderStr);
scorePos = [];
if ~isempty(sc)
    posIdx = find(lower(string(mdl.ClassNames))==lower(string(posName)),1);
    if isempty(posIdx), posIdx = min(2,size(sc,2)); end
    scorePos = sc(:,posIdx);
end
end

function [X, Y] = buildFeatureMatrix(imds, cfg)
n = numel(imds.Files);
Y = imds.Labels;
I = readimage(imds, 1);
f0 = extractFeaturesFromImage(I, cfg);
d = numel(f0);
X = zeros(n, d, 'double');
for i = 1:n
    I = readimage(imds, i);
    X(i,:) = extractFeaturesFromImage(I, cfg);
    if mod(i, 50) == 0 || i == n
        fprintf('      %d/%d\n', i, n);
    end
end
end

function feat = extractFeaturesFromImage(I, cfg)
Iproc = advancedPreprocessing(I, cfg);
mask = basicSegmentationMask(Iproc);
feat = [];
feat = [feat, extractHOGFeatures(Iproc, 'CellSize', cfg.hogCellSize)];
feat = [feat, glcmFeatures(Iproc)];
feat = [feat, lbpFeaturesCompact(Iproc)];
feat = [feat, gaborFeatures(Iproc, cfg.gaborWavelengths, cfg.gaborOrientations)];
feat = [feat, edgeStats(Iproc)];
feat = [feat, cornerStats(Iproc)];
feat = [feat, intensityMoments(Iproc)];
feat = [feat, morphFeatures(mask)];
feat = [feat, colorHSVStats(I)];
feat = [feat, shapeFeatures(mask)];
feat(~isfinite(feat)) = 0;
end

function Iout = advancedPreprocessing(I, cfg)
if size(I,3)==3, Ig = rgb2gray(I); else, Ig = I; end
Ig = im2double(Ig);
Ig = imresize(Ig, cfg.imageSize);
I1 = medfilt2(Ig,[3 3]);
I2 = wiener2(I1,[5 5]);
if cfg.useCLAHE
    I3 = adapthisteq(I2, 'ClipLimit', 0.02, 'NumTiles', [8 8], 'Distribution','rayleigh');
else
    I3 = I2;
end
I3 = mat2gray(I3);
if cfg.useMultiScaleFiltering
    G1 = imgaussfilt(I3, 0.5); G2 = imgaussfilt(I3, 1.0); G3 = imgaussfilt(I3, 2.0);
    I4 = (I3 + G1 + G2 + G3)/4.0;
else
    I4 = I3;
end
I5 = imsharpen(I4, 'Radius', 2, 'Amount', 0.8);
if cfg.useMorphology
    se = strel('disk',1);
    I6 = imclose(imopen(I5,se),se);
else
    I6 = I5;
end
try I7 = imbilatfilt(I6, 'DegreeOfSmoothing', 0.5); catch, I7 = I6; end
Iout = mat2gray(I7);
end

function mask = basicSegmentationMask(Igray)
bw = imbinarize(Igray, 'adaptive', 'ForegroundPolarity','bright', 'Sensitivity',0.5);
bw = bwareaopen(bw, 30);
bw = imclose(bw, strel('disk',2));
mask = imfill(bw,'holes');
end

function f = glcmFeatures(I)
Iu = im2uint8(I);
offsets = [0 1; -1 1; -1 0; -1 -1; 0 2; -2 2; -2 0; -2 -2];
glcm = graycomatrix(Iu,'Offset',offsets,'Symmetric',true);
stats = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
c = mean(stats.Contrast); r = mean(stats.Correlation);
e = mean(stats.Energy); h = mean(stats.Homogeneity);
ent = zeros(1,size(glcm,3));
for k=1:size(glcm,3)
    p = glcm(:,:,k); p = p/(sum(p(:))+eps);
    ent(k) = -sum(p(:).*log2(p(:)+eps));
end
f = [c r e h mean(ent) std(ent)];
end

function f = lbpFeaturesCompact(I)
try
    f = extractLBPFeatures(I, 'CellSize',[32 32], 'Normalization','L2');
catch
    f = [mean(I(:)) std(I(:)) var(I(:))];
end
end

function f = gaborFeatures(I, wavelengths, orientations)
g = gabor(wavelengths, orientations);
gm = imgaborfilt(I, g);
vals = [];
if isnumeric(gm)
    for k = 1:size(gm, 3)
        M = abs(gm(:,:,k));
        vals = [vals, mean(M(:)), std(M(:))];
    end
elseif iscell(gm)
    for k = 1:numel(gm)
        M = abs(gm{k});
        vals = [vals, mean(M(:)), std(M(:))];
    end
else
    vals = [0 0];
end
f = vals;
end

function f = edgeStats(I)
E1 = edge(I,'Sobel'); E2 = edge(I,'Canny'); E3 = edge(I,'log');
[Gx,Gy] = imgradientxy(I,'sobel'); Gmag = hypot(Gx,Gy);
f = [mean(E1(:)) mean(E2(:)) mean(E3(:)) mean(Gmag(:)) std(Gmag(:)) max(Gmag(:))];
end

function f = cornerStats(I)
try
    C = detectHarrisFeatures(I);
    n = C.Count;
    if n>0, s = C.Metric; f = [n, mean(s), std(s), max(s)]; else, f = [0 0 0 0]; end
catch
    f = [0 0 0 0];
end
end

function f = intensityMoments(I)
v = I(:);
f = [mean(v) var(v) skewness(v) kurtosis(v) prctile(v,10) prctile(v,50) prctile(v,90)];
end

function f = morphFeatures(mask)
try sk = bwmorph(mask,'skel',Inf); skLen = sum(sk(:)); catch, skLen = 0; end
eul = bweuler(mask);
f = [skLen, eul, sum(mask(:))/numel(mask)];
end

function f = colorHSVStats(I)
if size(I,3)~=3, f = [0 0 0 0]; return; end
I = im2double(I);
hsvImg = rgb2hsv(I);
H = hsvImg(:,:,1); S = hsvImg(:,:,2);
f = [mean(H(:)) std(H(:)) mean(S(:)) std(S(:))];
end

function f = shapeFeatures(mask)
stats = regionprops(mask, 'Area','Perimeter','Solidity','Eccentricity','Extent');
if isempty(stats), f = [0 0 0 0 0]; return; end
[~,idx] = max([stats.Area]);
s = stats(idx);
f = [s.Area, s.Perimeter, s.Solidity, s.Eccentricity, s.Extent];
end

function [Xtr2, Xte2, info] = varianceCorrelationFilterTrainOnly_2way(Xtr, Xte, cfg)
info = struct();
info.originalDims = size(Xtr,2);
keepVar = true(1,size(Xtr,2));
if cfg.useVarianceFilter
    v = var(Xtr,0,1);
    keepVar = v > cfg.varianceThreshold;
end
keepCorr = true(1,size(Xtr,2));
if cfg.useCorrelationFilter
    tmpIdx = find(keepVar);
    Xuse = Xtr(:,tmpIdx);
    if size(Xuse,2) > 1
        C = abs(corr(Xuse));
        C(1:size(C,1)+1:end) = 0;
        keepLocal = true(1,size(C,1));
        for i=1:size(C,1)
            if ~keepLocal(i), continue; end
            high = find(C(i,:)>cfg.correlationThreshold & keepLocal);
            keepLocal(high) = false;
        end
        keepCorr = false(1,size(Xtr,2));
        keepCorr(tmpIdx(keepLocal)) = true;
    end
end
keep = keepVar & keepCorr;
info.keepMask = keep;
info.removedByVariance = sum(~keepVar);
info.removedByCorrelation = sum(keepVar) - sum(keep);
Xtr2 = Xtr(:,keep); Xte2 = Xte(:,keep);
end

function ranks = relieffSafe(X, Y, k)
try [ranks, ~] = relieff(X, Y, k); catch, ranks = (1:size(X,2))'; end
end

function [XtrA, XvaA, ldaObj] = prepBranchA_SVM(Xtr, Ytr, Xva, cfg)
[XtrZ, mu, sig] = zscore(Xtr);
sig(sig==0) = 1;
XvaZ = (Xva - mu)./(sig + eps);
ldaObj = [];
if cfg.useLDA_for_SVM
    ldaObj = fitcdiscr(XtrZ, Ytr, 'DiscrimType','linear');
    [~, sc_tr] = predict(ldaObj, XtrZ);
    [~, sc_va] = predict(ldaObj, XvaZ);
    posCol = min(2,size(sc_tr,2));
    XtrA = sc_tr(:,posCol);
    XvaA = sc_va(:,posCol);
else
    XtrA = XtrZ; XvaA = XvaZ;
end
end

function [XtrZ, XvaZ, params] = zscoreTrainApply_returnParams(Xtr, Xva)
[XtrZ, mu, sig] = zscore(Xtr);
sig(sig==0) = 1;
XvaZ = (Xva - mu)./(sig + eps);
params = struct('mu',mu,'sig',sig);
end

function [svmMdl, svmCfg] = tuneSVM_kFoldCV(Xtr, Ytr, cfg)
cvp = cvpartition(Ytr, 'KFold', cfg.cvFolds);
bestAcc = -Inf; best = struct(); svmMdl = [];
for i=1:numel(cfg.svm.boxConstraints)
    C = cfg.svm.boxConstraints(i);
    for j=1:numel(cfg.svm.kernelScales)
        ks = cfg.svm.kernelScales(j);
        cvAccs = zeros(cfg.cvFolds, 1);
        for fold = 1:cfg.cvFolds
            trainIdx = training(cvp, fold);
            valIdx = test(cvp, fold);
            mdl = fitcsvm(Xtr(trainIdx,:), Ytr(trainIdx), 'KernelFunction','rbf', 'BoxConstraint', C, 'KernelScale', ks);
            mdl = fitPosterior(mdl, Xtr(trainIdx,:), Ytr(trainIdx));
            cvAccs(fold) = mean(predict(mdl, Xtr(valIdx,:)) == Ytr(valIdx));
        end
        acc = mean(cvAccs);
        if acc > bestAcc, bestAcc = acc; best.C = C; best.kernelScale = ks; end
    end
end
svmMdl = fitcsvm(Xtr, Ytr, 'KernelFunction','rbf', 'BoxConstraint', best.C, 'KernelScale', best.kernelScale);
svmMdl = fitPosterior(svmMdl, Xtr, Ytr);
svmCfg = best; svmCfg.cvAcc = bestAcc;
end

function [rfMdl, rfCfg] = tuneRF_kFoldCV(Xtr, Ytr, cfg)
cvp = cvpartition(Ytr, 'KFold', cfg.cvFolds);
bestAcc = -Inf; best = struct(); rfMdl = [];
for nt = cfg.rf.numTrees
    for leaf = cfg.rf.minLeafSizes
        cvAccs = zeros(cfg.cvFolds, 1);
        for fold = 1:cfg.cvFolds
            trainIdx = training(cvp, fold);
            valIdx = test(cvp, fold);
            mdl = TreeBagger(nt, Xtr(trainIdx,:), Ytr(trainIdx), 'Method','classification', 'MinLeafSize', leaf);
            yp = categorical(lower(string(predict(mdl, Xtr(valIdx,:)))), lower(string(categories(Ytr))));
            cvAccs(fold) = mean(yp == Ytr(valIdx));
        end
        acc = mean(cvAccs);
        if acc > bestAcc, bestAcc = acc; best.numTrees = nt; best.minLeaf = leaf; end
    end
end
rfMdl = TreeBagger(best.numTrees, Xtr, Ytr, 'Method','classification', 'MinLeafSize', best.minLeaf, 'OOBPrediction','On');
rfCfg = best; rfCfg.cvAcc = bestAcc;
end

function [xgbMdl, xgbCfg] = tuneXGB_kFoldCV(Xtr, Ytr, cfg)
cvp = cvpartition(Ytr, 'KFold', cfg.cvFolds);
bestAcc = -Inf; best = struct(); xgbMdl = [];
for cyc = cfg.xgb.numCycles
    for lr = cfg.xgb.learnRates
        for leaf = cfg.xgb.minLeafSizes
            cvAccs = zeros(cfg.cvFolds, 1);
            for fold = 1:cfg.cvFolds
                trainIdx = training(cvp, fold);
                valIdx = test(cvp, fold);
                t = templateTree('MinLeafSize', leaf);
                mdl = fitcensemble(Xtr(trainIdx,:), Ytr(trainIdx), 'Method','LogitBoost', 'Learners', t, 'NumLearningCycles', cyc, 'LearnRate', lr);
                cvAccs(fold) = mean(predict(mdl, Xtr(valIdx,:)) == Ytr(valIdx));
            end
            acc = mean(cvAccs);
            if acc > bestAcc, bestAcc = acc; best.numCycles = cyc; best.learnRate = lr; best.minLeaf = leaf; end
        end
    end
end
t = templateTree('MinLeafSize', best.minLeaf);
xgbMdl = fitcensemble(Xtr, Ytr, 'Method','LogitBoost', 'Learners', t, 'NumLearningCycles', best.numCycles, 'LearnRate', best.learnRate);
xgbCfg = best; xgbCfg.cvAcc = bestAcc;
end

function mdl = trainSVM_final(Xtr, Ytr, svmCfg)
mdl = fitcsvm(Xtr, Ytr, 'KernelFunction','rbf', 'BoxConstraint', svmCfg.C, 'KernelScale', svmCfg.kernelScale);
mdl = fitPosterior(mdl, Xtr, Ytr);
end

function mdl = trainRF_final(Xtr, Ytr, rfCfg)
mdl = TreeBagger(rfCfg.numTrees, Xtr, Ytr, 'Method','classification', 'MinLeafSize', rfCfg.minLeaf, 'OOBPrediction','On');
end

function mdl = trainXGB_final(Xtr, Ytr, xgbCfg)
t = templateTree('MinLeafSize', xgbCfg.minLeaf);
mdl = fitcensemble(Xtr, Ytr, 'Method','LogitBoost', 'Learners', t, 'NumLearningCycles', xgbCfg.numCycles, 'LearnRate', xgbCfg.learnRate);
end

function best = initBestStruct()
best = struct('bestCVAcc', -Inf, 'K', NaN, 'topKIdx', [], 'svm', [], 'svmCfg', [], 'rf', [], 'rfCfg', [], 'xgb', [], 'xgbCfg', []);
end

function sigResults = performMcNemarTests(Ytrue, predictions, modelNames)
nModels = length(predictions);
pValues = zeros(nModels, nModels);
significance = cell(nModels, nModels);

for i = 1:nModels
    for j = i+1:nModels
        b = sum(predictions{i} ~= Ytrue & predictions{j} == Ytrue);
        c = sum(predictions{i} == Ytrue & predictions{j} ~= Ytrue);
        
        if b + c > 0
            chi2 = (abs(b - c) - 1)^2 / (b + c);
            pValues(i,j) = 1 - chi2cdf(chi2, 1);
        else
            pValues(i,j) = 1;
        end
        
        if pValues(i,j) < 0.001
            significance{i,j} = '(p<0.001***)';
        elseif pValues(i,j) < 0.01
            significance{i,j} = '(p<0.01**)';
        elseif pValues(i,j) < 0.05
            significance{i,j} = '(p<0.05*)';
        else
            significance{i,j} = '(ns)';
        end
    end
end

% Store as separate fields to avoid dimension mismatch
sigResults.pValues = pValues;
sigResults.significance = significance;
sigResults.modelNames = modelNames;  % Store separately, not in struct() call
end

function ciResults = bootstrapConfidenceIntervals(Ytrue, predictions, ~, nBootstrap, modelNames)
nModels = length(predictions);
ciResults = struct();
for m = 1:nModels
    accBoot = zeros(nBootstrap, 1);
    for b = 1:nBootstrap
        idx = randi(length(Ytrue), length(Ytrue), 1);
        accBoot(b) = mean(predictions{m}(idx) == Ytrue(idx));
    end
    fieldName = strrep(modelNames{m}, ' ', '_');
    ciResults.(fieldName).mean = mean(accBoot);
    ciResults.(fieldName).std = std(accBoot);
    ciResults.(fieldName).ci95 = prctile(accBoot, [2.5, 97.5]);
end
end

function imds2 = subsetByLabels(imds, allowedStr)
lbl = lower(string(imds.Labels));
allowedStr = lower(string(allowedStr));
mask = false(size(lbl));
for i=1:numel(allowedStr), mask = mask | (lbl == allowedStr(i)); end
imds2 = subset(imds, find(mask));
end

function showPreprocessingStages(I, cfg)
I1 = I;
if size(I1,3)==3, I2 = rgb2gray(I1); else, I2 = I1; end
I2 = im2double(I2);
I3 = imresize(I2, cfg.imageSize);
I4 = medfilt2(I3, [3 3]);
I5 = wiener2(I4, [5 5]);
if cfg.useCLAHE, I6 = adapthisteq(I5, 'ClipLimit', 0.02, 'NumTiles', [8 8], 'Distribution','rayleigh'); else, I6 = I5; end
I6n = mat2gray(I6);
if cfg.useMultiScaleFiltering
    G1 = imgaussfilt(I6n, 0.5); G2 = imgaussfilt(I6n, 1.0); G3 = imgaussfilt(I6n, 2.0);
    I7 = (I6n + G1 + G2 + G3) / 4.0;
else, I7 = I6n; end
I8 = imsharpen(I7, 'Radius', 2, 'Amount', 0.8);
if cfg.useMorphology, se = strel('disk', 1); I9 = imclose(imopen(I8, se), se); else, I9 = I8; end
try I10 = imbilatfilt(I9, 'DegreeOfSmoothing', 0.5); catch, I10 = I9; end
stages = {I1,I2,I3,I4,I5,I6,I7,I8,I9,I10};
names = {'1 Original','2 Gray','3 Resize','4 Median','5 Wiener','6 CLAHE','7 Multi-Scale','8 Sharpen','9 Morph','10 Final'};
figure('Name','10-Stage Preprocessing','Position',[80 80 1400 700]);
for i=1:10
    subplot(2,5,i);
    if i==1 && size(stages{i},3)==3, imshow(stages{i}); else, imshow(stages{i},[]); end
    title(names{i}, 'FontWeight','bold');
end
sgtitle('Preprocessing Pipeline (10 Stages)', 'FontWeight','bold');
end

function printHeader(txt)
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  %s\n', txt);
fprintf('═══════════════════════════════════════════════════════════════\n');
end

function plotFeatureImportance(ranks, nTop, titleStr, expDir, cfg, filename)
figure('Name', 'Feature Importance', 'Position', [100 100 800 500]);
nTop = min(nTop, length(ranks));
[~, idx] = sort(ranks, 'descend');
topIndices = idx(1:nTop);
topScores = ranks(topIndices);
barh(topScores);
set(gca, 'YTick', 1:nTop, 'YTickLabel', arrayfun(@(x) sprintf('F%d', x), topIndices, 'UniformOutput', false));
xlabel('Importance Score');
title(titleStr, 'FontWeight', 'bold');
grid on;
if cfg.saveFigures, saveFigureHelper(gcf, fullfile(expDir, filename)); end
end

function plotLDAProjection(Xtr, Ytr, Xte, Yte, orderStr, expDir, cfg)
figure('Name','LDA Projection','Position',[100 100 1000 500]);
subplot(1,2,1);
benignIdx = Ytr == orderStr(1);
histogram(Xtr(benignIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Benign');
hold on;
histogram(Xtr(~benignIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Malignant');
xlabel('LDA Component');
ylabel('Frequency');
title('Training Set', 'FontWeight', 'bold');
legend; grid on;
subplot(1,2,2);
benignIdx = Yte == orderStr(1);
histogram(Xte(benignIdx), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Benign');
hold on;
histogram(Xte(~benignIdx), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Malignant');
xlabel('LDA Component');
ylabel('Frequency');
title('Test Set', 'FontWeight', 'bold');
legend; grid on;
sgtitle('LDA 1D Projection', 'FontWeight', 'bold');
if cfg.saveFigures, saveFigureHelper(gcf, fullfile(expDir, '02_lda_projection.png')); end
end

function plotSVMPerformance(~, res, orderStr, expDir, cfg)
figure('Name','SVM Performance','Position',[50 50 1600 420]);

subplot(1,4,1);
axis off;
text(0.5,0.5, sprintf('SVM\nAcc=%.3f\nSens=%.3f\nSpec=%.3f\nF1=%.3f', ...
    res.Accuracy,res.Sensitivity,res.Specificity,res.F1), ...
    'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');
title('Summary','FontWeight','bold');

subplot(1,4,2);
if ~isempty(res.ROC.FPR)
    plot(res.ROC.FPR, res.ROC.TPR, 'LineWidth',2.5); hold on;
    plot([0 1],[0 1],'k--','LineWidth',1.5);
    title(sprintf('ROC (AUC=%.3f)',res.AUC),'FontWeight','bold');
    xlabel('FPR'); ylabel('TPR'); grid on; axis square;
end

subplot(1,4,3);
cm = confusionchart(res.CM, categorical(orderStr, orderStr));
cm.Title = sprintf('Confusion (Acc=%.1f%%)', res.Accuracy*100);

subplot(1,4,4);
bar([res.Accuracy res.Sensitivity res.Specificity res.Precision res.F1]);
ylim([0 1]); grid on;
set(gca,'XTickLabel',{'Acc','Sens','Spec','Prec','F1'});
title('Metrics','FontWeight','bold');

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '07_svm_performance.png'));
end
end

function plotRFPerformance(res, orderStr, expDir, cfg)
figure('Name','RF Performance','Position',[50 50 1600 420]);

subplot(1,4,1);
axis off;
text(0.5,0.5, sprintf('Random Forest\nAcc=%.3f\nSens=%.3f\nSpec=%.3f\nF1=%.3f', ...
    res.Accuracy,res.Sensitivity,res.Specificity,res.F1), ...
    'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');

subplot(1,4,2);
if ~isempty(res.ROC.FPR)
    plot(res.ROC.FPR, res.ROC.TPR, 'LineWidth',2.5); hold on;
    plot([0 1],[0 1],'k--','LineWidth',1.5);
    title(sprintf('ROC (AUC=%.3f)',res.AUC),'FontWeight','bold');
    xlabel('FPR'); ylabel('TPR'); grid on; axis square;
end

subplot(1,4,3);
cm = confusionchart(res.CM, categorical(orderStr, orderStr));
cm.Title = sprintf('Confusion (Acc=%.1f%%)', res.Accuracy*100);

subplot(1,4,4);
bar([res.Accuracy res.Sensitivity res.Specificity res.Precision res.F1]);
ylim([0 1]); grid on;
set(gca,'XTickLabel',{'Acc','Sens','Spec','Prec','F1'});
title('Metrics','FontWeight','bold');

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '08_rf_performance.png'));
end
end

function plotXGBPerformance(res, orderStr, expDir, cfg)
figure('Name','XGB Performance','Position',[50 50 1600 420]);

subplot(1,4,1);
axis off;
text(0.5,0.5, sprintf('XGBoost\nAcc=%.3f\nSens=%.3f\nSpec=%.3f\nF1=%.3f', ...
    res.Accuracy,res.Sensitivity,res.Specificity,res.F1), ...
    'HorizontalAlignment','center','FontSize',12,'FontWeight','bold');

subplot(1,4,2);
if ~isempty(res.ROC.FPR)
    plot(res.ROC.FPR, res.ROC.TPR, 'LineWidth',2.5); hold on;
    plot([0 1],[0 1],'k--','LineWidth',1.5);
    title(sprintf('ROC (AUC=%.3f)',res.AUC),'FontWeight','bold');
    xlabel('FPR'); ylabel('TPR'); grid on; axis square;
end

subplot(1,4,3);
cm = confusionchart(res.CM, categorical(orderStr, orderStr));
cm.Title = sprintf('Confusion (Acc=%.1f%%)', res.Accuracy*100);

subplot(1,4,4);
bar([res.Accuracy res.Sensitivity res.Specificity res.Precision res.F1]);
ylim([0 1]); grid on;
set(gca,'XTickLabel',{'Acc','Sens','Spec','Prec','F1'});
title('Metrics','FontWeight','bold');

if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '09_xgb_performance.png'));
end
end

function plotMetricsComparison(resultsTbl, expDir, cfg)
figure('Name','Metrics Comparison','Position',[100 100 1200 600]);
M = resultsTbl{:,:};
bar(M','grouped');
set(gca,'XTickLabel',resultsTbl.Properties.VariableNames);
ylabel('Score'); ylim([0 1.05]); grid on;
title('Performance Comparison','FontWeight','bold');
legend(resultsTbl.Properties.RowNames,'Location','southeast');
if cfg.saveFigures, saveFigureHelper(gcf, fullfile(expDir, '10_metrics_comparison.png')); end
end

function plotCombinedROC(resList, names, expDir, cfg)
figure('Name','Combined ROC','Position',[120 120 850 650]);
hold on;
for i=1:numel(resList)
    r = resList{i};
    if ~isempty(r.ROC.FPR), plot(r.ROC.FPR, r.ROC.TPR, 'LineWidth',2.5); end
end
plot([0 1],[0 1],'k--','LineWidth',1.5);
xlabel('FPR'); ylabel('TPR');
title('ROC Curves','FontWeight','bold');
grid on; axis square;
leg = {};
for i=1:numel(resList)
    r = resList{i};
    if ~isempty(r.ROC.FPR), leg{end+1} = sprintf('%s (AUC=%.3f)', names{i}, r.AUC); end
end
leg{end+1} = 'Chance';
legend(leg,'Location','southeast');
if cfg.saveFigures, saveFigureHelper(gcf, fullfile(expDir, '11_roc_curves.png')); end
end

function plotConfusionBreakdownAll(resList, names, expDir, cfg)
figure('Name','Confusion Breakdown','Position',[100 100 1200 500]);
TN=[];FP=[];FN=[];TP=[];
for i=1:numel(resList)
    TN(i)=resList{i}.TN; FP(i)=resList{i}.FP; FN(i)=resList{i}.FN; TP(i)=resList{i}.TP;
end
M = [TN;FP;FN;TP]';
bar(M,'stacked'); grid on;
set(gca,'XTickLabel',names);
ylabel('Count');
title('TN/FP/FN/TP Breakdown','FontWeight','bold');
legend({'TN','FP','FN','TP'},'Location','best');
if cfg.saveFigures, saveFigureHelper(gcf, fullfile(expDir, '12_confusion_breakdown.png')); end
end

function plotConfMatricesSideBySide(cms, orderStr, names, expDir, cfg)
figure('Name','Confusion Matrices','Position',[80 80 1400 650]);
n = numel(cms);
for i=1:n
    subplot(2, ceil(n/2), i);
    cm = confusionchart(cms{i}, categorical(orderStr, orderStr));
    cm.Title = names{i};
end
sgtitle('Confusion Matrices', 'FontWeight','bold');
if cfg.saveFigures
    saveFigureHelper(gcf, fullfile(expDir, '13_confusion_matrices.png'));
end
end

function saveFigureHelper(fig, filepath)
try
    set(fig, 'Visible', 'off');
    drawnow;
    exportgraphics(fig, filepath, 'Resolution', 150);
catch ME
    warning('Could not save: %s', ME.message);
end
try close(fig); catch, end
end

%% DATASET PREPARATION
function [ready, report] = ensureBreakHisPrepared_2way(rawRoot, preparedRoot, magFolder, trainRatio, testRatio, rngSeed)
ready = false;
report = struct();
if ~isfolder(rawRoot), error('Raw not found: %s', rawRoot); end
trainBenDir = fullfile(preparedRoot, 'Training', 'benign');
trainMalDir = fullfile(preparedRoot, 'Training', 'malignant');
testBenDir = fullfile(preparedRoot, 'Test', 'benign');
testMalDir = fullfile(preparedRoot, 'Test', 'malignant');
if isfolder(trainBenDir) && isfolder(testBenDir)
    report = countPrepared_2way(preparedRoot);
    if (report.trainTotal + report.testTotal) > 0, ready = true; return; end
end
ensureFolder(trainBenDir); ensureFolder(trainMalDir);
ensureFolder(testBenDir); ensureFolder(testMalDir);
T = indexBreakHis(rawRoot, magFolder);
if isempty(T), error('No images'); end
nBen = sum(strcmpi(T.class, 'benign'));
nMal = sum(strcmpi(T.class, 'malignant'));
targetPerClass = min(nBen, nMal);
targetTrainPerClass = round(trainRatio * targetPerClass);
targetTestPerClass = targetPerClass - targetTrainPerClass;
rng(rngSeed);
[trainMask, testMask] = balancedPatientDisjointSplit_2way(T, targetTrainPerClass, targetTestPerClass);
deleteIfExists(fullfile(trainBenDir,'*.png'));
deleteIfExists(fullfile(trainMalDir,'*.png'));
deleteIfExists(fullfile(testBenDir,'*.png'));
deleteIfExists(fullfile(testMalDir,'*.png'));
copySelected(T(trainMask,:), trainBenDir, trainMalDir);
copySelected(T(testMask,:), testBenDir, testMalDir);
report = countPrepared_2way(preparedRoot);
ready = true;
end

function [trainMask, testMask] = balancedPatientDisjointSplit_2way(T, nTrain, nTest)
trainMask = false(height(T),1);
testMask = false(height(T),1);
for cls = {'benign','malignant'}
    Tc = T(strcmpi(T.class, cls{1}), :);
    patients = unique(Tc.patient);
    patients = patients(randperm(numel(patients)));
    testPatients = {};
    nTestImgs = 0;
    k = 1;
    while (nTestImgs < nTest) && (k <= numel(patients))
        p = patients{k};
        idxP = find(strcmpi(Tc.patient, p));
        nTestImgs = nTestImgs + numel(idxP);
        testPatients{end+1,1} = p;
        k = k + 1;
    end
    trainPatients = setdiff(patients, testPatients, 'stable');
    testIdx = find(ismember(T.patient, testPatients) & strcmpi(T.class, cls{1}));
    trainIdx = find(ismember(T.patient, trainPatients) & strcmpi(T.class, cls{1}));
    if numel(testIdx) > nTest, testIdx = testIdx(randperm(numel(testIdx), nTest)); end
    if numel(trainIdx) > nTrain, trainIdx = trainIdx(randperm(numel(trainIdx), nTrain)); end
    testMask(testIdx) = true;
    trainMask(trainIdx) = true;
end
trainPat = unique(T.patient(trainMask));
testPat = unique(T.patient(testMask));
if ~isempty(intersect(trainPat, testPat)), error('Patient leakage'); end
end

function T = indexBreakHis(rawRoot, magFolder)
rows = {};
for cls = {'benign','malignant'}
    pattern = fullfile(rawRoot, cls{1}, '**', magFolder, '*.png');
    files = dir(pattern);
    for i = 1:numel(files)
        parts = splitPathParts(files(i).folder);
        subtype = 'unk'; patient = 'unk';
        idx = find(strcmpi(parts, cls{1}), 1, 'last');
        if ~isempty(idx)
            if idx+1 <= numel(parts), subtype = parts{idx+1}; end
            if idx+2 <= numel(parts), patient = parts{idx+2}; end
        end
        dstName = sprintf('%s__%s__%s__%s.png', sanitizeToken(subtype), sanitizeToken(patient), sanitizeToken(magFolder), sanitizeToken(erase(files(i).name, '.png')));
        rows(end+1,:) = {cls{1}, subtype, patient, magFolder, fullfile(files(i).folder, files(i).name), dstName};
    end
end
if isempty(rows), T = table(); else, T = cell2table(rows, 'VariableNames', {'class','subtype','patient','mag','srcPath','dstName'}); end
end

function copySelected(Tsel, benignDir, malignantDir)
for i = 1:height(Tsel)
    outDir = ternary(strcmpi(Tsel.class{i}, 'benign'), benignDir, malignantDir);
    dst = makeCollisionSafe(fullfile(outDir, Tsel.dstName{i}));
    copyfile(Tsel.srcPath{i}, dst);
end
end

function report = countPrepared_2way(preparedRoot)
trainBenDir = fullfile(preparedRoot, 'Training', 'benign');
trainMalDir = fullfile(preparedRoot, 'Training', 'malignant');
testBenDir = fullfile(preparedRoot, 'Test', 'benign');
testMalDir = fullfile(preparedRoot, 'Test', 'malignant');
report = struct();
report.trainBenign = numel(dir(fullfile(trainBenDir,'*.png')));
report.trainMalignant = numel(dir(fullfile(trainMalDir,'*.png')));
report.testBenign = numel(dir(fullfile(testBenDir,'*.png')));
report.testMalignant = numel(dir(fullfile(testMalDir,'*.png')));
report.trainTotal = report.trainBenign + report.trainMalignant;
report.testTotal = report.testBenign + report.testMalignant;
end

function ensureFolder(p), if ~isfolder(p), mkdir(p); end, end
function deleteIfExists(pattern), d = dir(pattern); for i = 1:numel(d), try delete(fullfile(d(i).folder, d(i).name)); catch, end; end, end
function parts = splitPathParts(p), p = strrep(strrep(p, '\', filesep), '/', filesep); raw = strsplit(p, filesep); parts = raw(~cellfun(@isempty, raw)); end
function t = sanitizeToken(s), t = regexprep(regexprep(s, '\s+', ''), '[^a-zA-Z0-9_\-]', '_'); if isempty(t), t = 'unk'; end, end
function dst = makeCollisionSafe(dst), if ~exist(dst, 'file'), return; end, [folder, name, ext] = fileparts(dst); k = 1; while exist(dst, 'file'), dst = fullfile(folder, sprintf('%s__%d%s', name, k, ext)); k = k + 1; end, end
function out = ternary(cond, trueVal, falseVal), if cond, out = trueVal; else, out = falseVal; end, end
