function breastCancerHistopathologyGui(modelMatPath)
% BREASTCANCERHISTOPATHOLOGYGUI
% Interactive GUI for BreastCancerHistopathological pipeline inference.
%
% Usage:
%   breastCancerHistopathologyGui
%   breastCancerHistopathologyGui('Training_results/models.mat')
%   breastCancerHistopathologyGui('Training_results/results.mat')
%
% Notes:
% - Expects results/models saved by BreastCancerHistopathological.m:
%   `results.mat` (contains results + trained) or `models*.mat` with a nearby results.mat.
% - When no model path is provided, auto-loads all `models*.mat` in Training_results (if found).
% - For accurate inference the GUI needs cfg, fsInfo.keepMask, featureIndices,
%   and zParams (plus ldaModel when LDA is enabled for SVM).

    if nargin < 1 || isempty(modelMatPath)
        modelMatPath = '';
    end

    app = struct();
    app.currentImage = [];
    app.currentImagePath = '';
    app.modelSets = struct('trained', {}, 'results', {}, 'cfg', {}, 'fsInfo', {}, ...
        'orderStr', {}, 'path', {}, 'resultsPath', {}, 'displayName', {}, 'metrics', {});
    app.activeModelIdx = 0;
    app.model = struct();
    app.bundleReady = false;
    app.uiUpdating = false;
    app.results = [];
    app.resultsPath = '';
    app.resultsFiles = strings(0, 1);
    app.resultsDisplayNames = strings(0, 1);
    app.projectRoot = fileparts(mfilename('fullpath'));

    buildUi();
    refreshModelSetUi();
    refreshResultsTab();
    refreshGraphTab();
    refreshTrainingLogTab();
    try
        loadDefaultModels(modelMatPath);
    catch me
        logLine("Failed to load model files: " + string(me.message));
    end
    try
        loadDefaultResults();
    catch me
        logLine("Failed to load results file: " + string(me.message));
    end

    %% ======================== UI BUILD ========================
    function buildUi()
        app.fig = uifigure('Name', 'Breast Cancer Histopathology Classifier', ...
            'Position', [100 100 1200 720]);

        root = uigridlayout(app.fig, [1 1]);
        root.ColumnWidth = {'1x'};
        root.RowHeight = {'1x'};
        root.Padding = [10 10 10 10];

        app.ui.tabGroup = uitabgroup(root);
        app.ui.tabGroup.Layout.Row = 1;
        app.ui.tabGroup.Layout.Column = 1;

        app.ui.tabTesting = uitab(app.ui.tabGroup, 'Title', 'Models');
        app.ui.tabResults = uitab(app.ui.tabGroup, 'Title', 'Training Results');
        app.ui.tabGraph = uitab(app.ui.tabGroup, 'Title', 'Training Graphs');
        app.ui.tabTrainingLog = uitab(app.ui.tabGroup, 'Title', 'Training logs');

        testingTabRoot = uigridlayout(app.ui.tabTesting, [2 1]);
        testingTabRoot.ColumnWidth = {'1x'};
        testingTabRoot.RowHeight = {'1x', 180};
        testingTabRoot.Padding = [8 8 8 8];

        testingRoot = uigridlayout(testingTabRoot, [1 2]);
        testingRoot.ColumnWidth = {360, '1x'};
        testingRoot.RowHeight = {'1x'};
        testingRoot.Padding = [0 0 0 0];
        testingRoot.Layout.Row = 1;
        testingRoot.Layout.Column = 1;

        leftPanel = uipanel(testingRoot, 'Title', 'Controls');
        leftPanel.Layout.Row = 1;
        leftPanel.Layout.Column = 1;

        left = uigridlayout(leftPanel, [2 1]);
        left.RowHeight = {'fit', '1x'};
        left.Padding = [8 8 8 8];

        % Model section
        modelPanel = uipanel(left, 'Title', 'Model');
        modelGrid = uigridlayout(modelPanel, [4 3]);
        modelGrid.ColumnWidth = {'fit', '1x', 'fit'};
        modelGrid.RowHeight = {'fit', 'fit', 'fit', 'fit'};
        modelGrid.Padding = [8 8 8 8];

        lbl = uilabel(modelGrid, 'Text', 'Model set:');
        lbl.Layout.Row = 1;
        lbl.Layout.Column = 1;
        app.ui.modelSetDropdown = uidropdown(modelGrid, ...
            'Items', {'(none)'}, 'ItemsData', 0, 'Value', 0, ...
            'ValueChangedFcn', @(~,~)onModelSetChanged());
        app.ui.modelSetDropdown.Layout.Row = 1;
        app.ui.modelSetDropdown.Layout.Column = [2 3];

        lbl = uilabel(modelGrid, 'Text', 'Algorithms:');
        lbl.Layout.Row = 2;
        lbl.Layout.Column = 1;
        algGrid = uigridlayout(modelGrid, [1 3]);
        algGrid.Layout.Row = 2;
        algGrid.Layout.Column = [2 3];
        algGrid.RowHeight = {'fit'};
        algGrid.ColumnWidth = {'fit', 'fit', 'fit'};
        algGrid.Padding = [0 0 0 0];
        app.ui.chkSVM = uicheckbox(algGrid, 'Text', 'SVM', 'Value', true, ...
            'ValueChangedFcn', @(~,~)onAlgorithmSelectionChanged());
        app.ui.chkRF = uicheckbox(algGrid, 'Text', 'Random Forest', 'Value', false, ...
            'ValueChangedFcn', @(~,~)onAlgorithmSelectionChanged());
        app.ui.chkXGB = uicheckbox(algGrid, 'Text', 'XGBoost', 'Value', false, ...
            'ValueChangedFcn', @(~,~)onAlgorithmSelectionChanged());

        app.ui.lblBundle = uilabel(modelGrid, 'Text', 'Bundle: -', ...
            'FontColor', [0.5 0.5 0.5]);
        app.ui.lblBundle.Layout.Row = 3;
        app.ui.lblBundle.Layout.Column = [1 3];
        app.ui.lblBundle.Tooltip = 'Bundle readiness for inference.';

        app.ui.lblOverall = uilabel(modelGrid, 'Text', 'Overall: -', ...
            'FontColor', [0.25 0.25 0.25], 'FontSize', 11);
        app.ui.lblOverall.Layout.Row = 4;
        app.ui.lblOverall.Layout.Column = [1 3];

        % Image section
        imgPanel = uipanel(left, 'Title', 'Image');
        imgGrid = uigridlayout(imgPanel, [2 3]);
        imgGrid.ColumnWidth = {'fit', '1x', 'fit'};
        imgGrid.RowHeight = {'fit', 'fit'};
        imgGrid.Padding = [8 8 8 8];

        app.ui.btnUpload = uibutton(imgGrid, 'Text', 'Upload...', ...
            'ButtonPushedFcn', @(~,~)onUploadImage());
        app.ui.btnUpload.Layout.Column = [1 3];

        uilabel(imgGrid, 'Text', 'Path:');
        app.ui.imagePath = uieditfield(imgGrid, 'text', 'Editable', 'off');
        app.ui.btnClear = uibutton(imgGrid, 'Text', 'Clear', ...
            'ButtonPushedFcn', @(~,~)onClearImage());

        % Right side (preview + scores)
        rightPanel = uipanel(testingRoot, 'Title', 'Preview');
        rightPanel.Layout.Row = 1;
        rightPanel.Layout.Column = 2;
        right = uigridlayout(rightPanel, [1 2]);
        right.ColumnWidth = {'1x', '1x'};
        right.RowHeight = {'1x'};
        right.Padding = [8 8 8 8];

        imgPanelRight = uipanel(right, 'Title', 'Image');
        imgPanelRight.Layout.Row = 1;
        imgPanelRight.Layout.Column = 1;
        imgGridRight = uigridlayout(imgPanelRight, [1 1]);
        imgGridRight.RowHeight = {'1x'};
        imgGridRight.ColumnWidth = {'1x'};
        imgGridRight.Padding = [0 0 0 0];

        app.ui.axImage = uiaxes(imgGridRight);
        title(app.ui.axImage, 'Uploaded Image');
        app.ui.axImage.XTick = [];
        app.ui.axImage.YTick = [];

        scoresPanel = uipanel(right, 'Title', 'Scores');
        scoresPanel.Layout.Row = 1;
        scoresPanel.Layout.Column = 2;
        scoresGrid = uigridlayout(scoresPanel, [2 1]);
        scoresGrid.RowHeight = {'2x', '1x'};
        scoresGrid.Padding = [8 8 8 8];
        app.ui.axCompare = uiaxes(scoresGrid);
        app.ui.axCompare.Layout.Row = 1;
        title(app.ui.axCompare, 'Algorithm Comparison');
        app.ui.tblCompare = uitable(scoresGrid, ...
            'Data', cell(0, 5), ...
            'ColumnName', {'Algorithm', 'Label', 'Conf', 'Benign', 'Malignant'}, ...
            'RowName', {});
        app.ui.tblCompare.Layout.Row = 2;
        try
            app.ui.tblCompare.ColumnEditable = false(1, 5);
        catch
        end
        try
            app.ui.tblCompare.RowStriping = 'on';
        catch
        end
        try
            app.ui.tblCompare.FontSize = 12;
        catch
        end

        % Results tab
        resultsRoot = uigridlayout(app.ui.tabResults, [3 1]);
        resultsRoot.RowHeight = {'fit', '3x', '1.6x'};
        resultsRoot.ColumnWidth = {'1x'};
        resultsRoot.Padding = [4 8 8 8];

        resultsPanel = uipanel(resultsRoot, 'Title', 'Results File');
        resultsPanel.Layout.Row = 1;
        resultsPanel.Layout.Column = 1;
        resultsGrid = uigridlayout(resultsPanel, [1 3]);
        resultsGrid.ColumnWidth = {'fit', '1x', 'fit'};
        resultsGrid.RowHeight = {'fit'};
        resultsGrid.Padding = [8 8 8 8];

        lbl = uilabel(resultsGrid, 'Text', 'Results set:');
        lbl.Layout.Row = 1;
        lbl.Layout.Column = 1;
        app.ui.resultsDropdown = uidropdown(resultsGrid, ...
            'Items', {'(none)'}, 'ItemsData', 0, 'Value', 0, ...
            'ValueChangedFcn', @(~,~)onResultsSetChanged());
        app.ui.resultsDropdown.Layout.Row = 1;
        app.ui.resultsDropdown.Layout.Column = 2;
        app.ui.btnExportResults = uibutton(resultsGrid, 'Text', 'Export CSV...', ...
            'ButtonPushedFcn', @(~,~)onExportResults());
        app.ui.btnExportResults.Layout.Row = 1;
        app.ui.btnExportResults.Layout.Column = 3;

        resultsMain = uigridlayout(resultsRoot, [2 2]);
        resultsMain.RowHeight = {'1x', '1x'};
        resultsMain.ColumnWidth = {'1x', '1x'};
        resultsMain.Layout.Row = 2;
        resultsMain.Layout.Column = 1;
        resultsMain.Padding = [0 0 0 0];

        testPanel = uipanel(resultsMain, 'Title', 'Test Results');
        testPanel.Layout.Row = 1;
        testPanel.Layout.Column = 1;
        testGrid = uigridlayout(testPanel, [1 1]);
        testGrid.RowHeight = {'1x'};
        testGrid.ColumnWidth = {'1x'};
        testGrid.Padding = [0 0 0 0];
        app.ui.tblTestResults = uitable(testGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblTestResults.Layout.Row = 1;
        app.ui.tblTestResults.Layout.Column = 1;

        metricsPanel = uipanel(resultsMain, 'Title', 'Metrics Summary');
        metricsPanel.Layout.Row = 1;
        metricsPanel.Layout.Column = 2;
        metricsGrid = uigridlayout(metricsPanel, [1 1]);
        metricsGrid.RowHeight = {'1x'};
        metricsGrid.ColumnWidth = {'1x'};
        metricsGrid.Padding = [0 0 0 0];
        app.ui.tblMetrics = uitable(metricsGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblMetrics.Layout.Row = 1;
        app.ui.tblMetrics.Layout.Column = 1;

        confPanel = uipanel(resultsMain, 'Title', 'Confusion Matrices');
        confPanel.Layout.Row = 2;
        confPanel.Layout.Column = 1;
        confGrid = uigridlayout(confPanel, [1 3]);
        confGrid.RowHeight = {'1x'};
        confGrid.ColumnWidth = {'1x', '1x', '1x'};
        confGrid.Padding = [8 8 8 8];

        confSvmPanel = uipanel(confGrid, 'Title', 'SVM');
        confSvmPanel.Layout.Row = 1;
        confSvmPanel.Layout.Column = 1;
        confSvmGrid = uigridlayout(confSvmPanel, [1 1]);
        confSvmGrid.RowHeight = {'1x'};
        confSvmGrid.ColumnWidth = {'1x'};
        confSvmGrid.Padding = [0 0 0 0];
        app.ui.tblConfSVM = uitable(confSvmGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblConfSVM.Layout.Row = 1;
        app.ui.tblConfSVM.Layout.Column = 1;

        confRfPanel = uipanel(confGrid, 'Title', 'RF');
        confRfPanel.Layout.Row = 1;
        confRfPanel.Layout.Column = 2;
        confRfGrid = uigridlayout(confRfPanel, [1 1]);
        confRfGrid.RowHeight = {'1x'};
        confRfGrid.ColumnWidth = {'1x'};
        confRfGrid.Padding = [0 0 0 0];
        app.ui.tblConfRF = uitable(confRfGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblConfRF.Layout.Row = 1;
        app.ui.tblConfRF.Layout.Column = 1;

        confXgbPanel = uipanel(confGrid, 'Title', 'XGB');
        confXgbPanel.Layout.Row = 1;
        confXgbPanel.Layout.Column = 3;
        confXgbGrid = uigridlayout(confXgbPanel, [1 1]);
        confXgbGrid.RowHeight = {'1x'};
        confXgbGrid.ColumnWidth = {'1x'};
        confXgbGrid.Padding = [0 0 0 0];
        app.ui.tblConfXGB = uitable(confXgbGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblConfXGB.Layout.Row = 1;
        app.ui.tblConfXGB.Layout.Column = 1;

        rocPanel = uipanel(resultsMain, 'Title', 'ROC Curves');
        rocPanel.Layout.Row = 2;
        rocPanel.Layout.Column = 2;
        rocGrid = uigridlayout(rocPanel, [1 3]);
        rocGrid.RowHeight = {'1x'};
        rocGrid.ColumnWidth = {'1x', '1x', '1x'};
        rocGrid.Padding = [8 8 8 8];

        app.ui.axRocSVM = uiaxes(rocGrid);
        app.ui.axRocSVM.Layout.Row = 1;
        app.ui.axRocSVM.Layout.Column = 1;
        title(app.ui.axRocSVM, 'SVM');

        app.ui.axRocRF = uiaxes(rocGrid);
        app.ui.axRocRF.Layout.Row = 1;
        app.ui.axRocRF.Layout.Column = 2;
        title(app.ui.axRocRF, 'RF');

        app.ui.axRocXGB = uiaxes(rocGrid);
        app.ui.axRocXGB.Layout.Row = 1;
        app.ui.axRocXGB.Layout.Column = 3;
        title(app.ui.axRocXGB, 'XGB');

        cfgTimingPanel = uipanel(resultsRoot, 'Title', 'Config and Timing');
        cfgTimingPanel.Layout.Row = 3;
        cfgTimingPanel.Layout.Column = 1;
        cfgTimingGrid = uigridlayout(cfgTimingPanel, [1 2]);
        cfgTimingGrid.RowHeight = {'1x'};
        cfgTimingGrid.ColumnWidth = {'1x', '1x'};
        cfgTimingGrid.Padding = [8 8 8 8];

        cfgPanel = uipanel(cfgTimingGrid, 'Title', 'Config (cfg)');
        cfgPanel.Layout.Row = 1;
        cfgPanel.Layout.Column = 1;
        cfgGrid = uigridlayout(cfgPanel, [1 1]);
        cfgGrid.RowHeight = {'1x'};
        cfgGrid.ColumnWidth = {'1x'};
        cfgGrid.Padding = [0 0 0 0];
        app.ui.tblCfg = uitable(cfgGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblCfg.Layout.Row = 1;
        app.ui.tblCfg.Layout.Column = 1;

        timingPanel = uipanel(cfgTimingGrid, 'Title', 'Timing');
        timingPanel.Layout.Row = 1;
        timingPanel.Layout.Column = 2;
        timingGrid = uigridlayout(timingPanel, [1 1]);
        timingGrid.RowHeight = {'1x'};
        timingGrid.ColumnWidth = {'1x'};
        timingGrid.Padding = [0 0 0 0];
        app.ui.tblTiming = uitable(timingGrid, 'Data', cell(0, 0), ...
            'ColumnName', {}, 'RowName', {});
        app.ui.tblTiming.Layout.Row = 1;
        app.ui.tblTiming.Layout.Column = 1;

        % Graph tab
        graphRoot = uigridlayout(app.ui.tabGraph, [1 1]);
        graphRoot.RowHeight = {'1x'};
        graphRoot.ColumnWidth = {'1x'};
        graphRoot.Padding = [8 8 8 8];

        app.ui.graphPanel = uipanel(graphRoot, 'Title', 'Training_results Images');
        app.ui.graphPanel.Layout.Row = 1;
        app.ui.graphPanel.Layout.Column = 1;
        try
            app.ui.graphPanel.Scrollable = 'on';
        catch
        end

        % Training log tab
        trainingLogRoot = uigridlayout(app.ui.tabTrainingLog, [1 1]);
        trainingLogRoot.RowHeight = {'1x'};
        trainingLogRoot.ColumnWidth = {'1x'};
        trainingLogRoot.Padding = [8 8 8 8];

        app.ui.trainingLogText = uitextarea(trainingLogRoot, 'Editable', 'off');
        app.ui.trainingLogText.Layout.Row = 1;
        app.ui.trainingLogText.Layout.Column = 1;
        try
            app.ui.trainingLogText.WordWrap = 'on';
        catch
        end
        try
            app.ui.trainingLogText.FontName = 'Menlo';
            app.ui.trainingLogText.FontSize = 12;
        catch
        end

        % Log (Models tab only)
        logPanel = uipanel(testingTabRoot, 'Title', 'Log');
        logPanel.Layout.Row = 2;
        logPanel.Layout.Column = 1;
        logGrid = uigridlayout(logPanel, [1 1]);
        logGrid.RowHeight = {'1x'};
        logGrid.ColumnWidth = {'1x'};
        logGrid.Padding = [0 0 0 0];
        app.ui.log = uitextarea(logGrid, 'Editable', 'off');
        app.ui.log.Layout.Row = 1;
        app.ui.log.Layout.Column = 1;
        try
            app.ui.log.WordWrap = 'on';
        catch
        end
        try
            app.ui.log.FontName = 'Menlo';
            app.ui.log.FontSize = 12;
        catch
        end
    end

    %% ======================== UI CALLBACKS ========================
    function onAddModel()
        [file, path] = uigetfile( ...
            {'models*.mat', 'Model files (models*.mat)'; ...
             'results*.mat', 'Results files (results*.mat)'; ...
             '*.mat', 'MAT-files (*.mat)'}, ...
            'Select model .mat file');
        if isequal(file, 0)
            return;
        end
        addModelFile(fullfile(path, file), true);
    end

    function onModelSetChanged()
        if app.uiUpdating || isempty(app.modelSets)
            return;
        end
        idx = app.ui.modelSetDropdown.Value;
        if isempty(idx) || ~isnumeric(idx) || idx < 1 || idx > numel(app.modelSets) || idx == app.activeModelIdx
            return;
        end
        setActiveModelIdx(idx);
    end

    function onResultsSetChanged()
        if app.uiUpdating || isempty(app.resultsFiles)
            return;
        end
        idx = app.ui.resultsDropdown.Value;
        if isempty(idx) || ~isnumeric(idx) || idx < 1 || idx > numel(app.resultsFiles)
            return;
        end
        try
            loadResultsFile(app.resultsFiles(idx));
        catch me
            logLine("Load results failed: " + string(me.message));
        end
    end

    function onExportResults()
        if isempty(app.results)
            logLine('Load results first.');
            return;
        end
        [file, path] = uiputfile('results_export.csv', 'Save results CSV');
        if isequal(file, 0)
            return;
        end
        try
            exportResultsCsv(fullfile(path, file));
        catch me
            logLine("Export results failed: " + string(me.message));
        end
    end

    function onUploadImage()
        [file, path] = uigetfile( ...
            {'*.png;*.jpg;*.jpeg;*.tif;*.tiff', 'Images (*.png, *.jpg, *.tif)'; '*.*', 'All Files'}, ...
            'Select an image');
        if isequal(file, 0)
            return;
        end

        imgPath = fullfile(path, file);
        try
            I = imread(imgPath);
        catch me
            logLine("Upload error: " + string(me.message));
            try
                uialert(app.fig, char(string(me.message)), 'Upload Error');
            catch
            end
            return;
        end

        app.currentImage = I;
        app.currentImagePath = imgPath;
        app.ui.imagePath.Value = imgPath;
        showImage(I);
        logLine("Loaded image: " + string(imgPath));

        if app.bundleReady
            onPredict();
        end
    end

    function onClearImage()
        app.currentImage = [];
        app.currentImagePath = '';
        app.ui.imagePath.Value = '';
        cla(app.ui.axImage);
        if isfield(app.ui, 'axCompare') && ~isempty(app.ui.axCompare)
            cla(app.ui.axCompare);
        end
        if isfield(app.ui, 'tblCompare') && ~isempty(app.ui.tblCompare)
            app.ui.tblCompare.Data = cell(0, 5);
        end
        logLine('Cleared image.');
    end

    function onAlgorithmSelectionChanged()
        if app.uiUpdating
            return;
        end
        if ~isempty(app.currentImage) && app.bundleReady
            onPredict();
        end
    end

    function onPredict()
        try
            if isempty(app.currentImage)
                logLine('Upload an image first.');
                return;
            end
            if ~app.bundleReady
                logLine('Load a results/models file with config before predicting.');
                return;
            end

            algs = getSelectedAlgorithms();
            if isempty(algs)
                logLine('Select at least one algorithm.');
                if isfield(app.ui, 'tblCompare') && ~isempty(app.ui.tblCompare)
                    app.ui.tblCompare.Data = cell(0, 5);
                end
                if isfield(app.ui, 'axCompare') && ~isempty(app.ui.axCompare)
                    cla(app.ui.axCompare);
                end
                return;
            end

            results = predictAlgorithmsList(app.currentImage, algs);
            isCompare = numel(algs) > 1;

            best = showCompareResults(results);
            if isfield(best, 'ok') && best.ok
                if isCompare
                    logLine("Compare: best=" + best.algorithm + " label=" + best.label + " conf=" + fmtScore(best.conf));
                else
                    logLine("Predicted with " + best.algorithm + ": " + best.label + " (" + fmtScore(best.conf) + ")");
                end
            else
                if isstruct(results) && numel(results) >= 1 && isfield(results(1), 'message') && strlength(results(1).message) > 0
                    logLine("Predict failed: " + results(1).message);
                else
                    logLine("Predict failed.");
                end
            end
        catch me
            logLine("Predict error: " + string(me.message));
            try
                uialert(app.fig, char(string(me.message)), 'Predict Error');
            catch
            end
        end
    end

    %% ======================== CORE LOGIC ========================
    function addModelFile(pathToMat, selectAfter)
        if nargin < 2
            selectAfter = true;
        end

        pathToMat = normalizeModelPath(pathToMat);
        [trained, results, resultsPath] = loadModelBundle(pathToMat);
        if isempty(trained)
            error('Missing "trained" in %s (load models.mat or results.mat).', pathToMat);
        end

        entry = buildModelEntry(pathToMat, trained, results, resultsPath);

        existingIdx = find(strcmp(string(pathToMat), string({app.modelSets.path})), 1);
        if isempty(existingIdx)
            app.modelSets(end+1) = entry; %#ok<AGROW>
            existingIdx = numel(app.modelSets);
        else
            app.modelSets(existingIdx) = entry;
        end

        logLine("Loaded model file: " + string(pathToMat));

        if selectAfter
            setActiveModelIdx(existingIdx);
        else
            refreshModelSetUi();
        end
    end

    function attachResultsFile(resultsPath)
        if isempty(app.modelSets) || app.activeModelIdx < 1
            return;
        end
        resultsPath = normalizeModelPath(resultsPath);
        s = load(resultsPath);
        if ~isfield(s, 'results')
            error('No "results" struct found in %s.', resultsPath);
        end

        entry = app.modelSets(app.activeModelIdx);
        entry.results = s.results;
        entry.resultsPath = resultsPath;
        if isfield(s, 'trained') && isempty(entry.trained)
            entry.trained = s.trained;
        end
        entry = applyResultsMetadata(entry);
        app.modelSets(app.activeModelIdx) = entry;
        setActiveModelIdx(app.activeModelIdx);
        logLine("Attached results: " + string(resultsPath));
    end

    function setActiveModelIdx(idx)
        if isempty(app.modelSets)
            app.activeModelIdx = 0;
            app.model = struct();
            refreshModelSetUi();
            return;
        end

        idx = max(1, min(idx, numel(app.modelSets)));
        app.activeModelIdx = idx;
        app.model = app.modelSets(idx);

        app.uiUpdating = true;
        refreshModelSetUi();

        updateBundleStatus();
        updateOverallStatus();
        syncResultsTabFromModel();
        app.uiUpdating = false;

        if ~isempty(app.currentImage) && app.bundleReady
            onPredict();
        end
    end

    function refreshModelSetUi()
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'modelSetDropdown') || isempty(app.ui.modelSetDropdown)
            return;
        end

        wasUpdating = app.uiUpdating;

        if isempty(app.modelSets)
            app.uiUpdating = true;
            app.ui.modelSetDropdown.Items = {'(none)'};
            app.ui.modelSetDropdown.ItemsData = 0;
            app.ui.modelSetDropdown.Value = 0;
            app.ui.modelSetDropdown.Enable = 'off';
            setAlgorithmCheckboxesEnabled(false);
            app.ui.lblBundle.Text = 'Bundle: -';
            app.ui.lblBundle.FontColor = [0.5 0.5 0.5];
            if isfield(app.ui, 'lblOverall') && ~isempty(app.ui.lblOverall)
                app.ui.lblOverall.Text = 'Overall: -';
                app.ui.lblOverall.FontColor = [0.5 0.5 0.5];
            end
            app.uiUpdating = wasUpdating;
            return;
        end

        if app.activeModelIdx < 1 || app.activeModelIdx > numel(app.modelSets)
            app.activeModelIdx = 1;
        end

        app.uiUpdating = true;
        app.ui.modelSetDropdown.Enable = 'on';
        nModels = numel(app.modelSets);
        labels = "Model " + (1:nModels) + ": " + string({app.modelSets.displayName});
        app.ui.modelSetDropdown.Items = cellstr(labels);
        app.ui.modelSetDropdown.ItemsData = 1:nModels;
        app.ui.modelSetDropdown.Value = app.activeModelIdx;
        setAlgorithmCheckboxesEnabled(true);
        app.uiUpdating = wasUpdating;
    end

    function setAlgorithmCheckboxesEnabled(isEnabled)
        if isfield(app.ui, 'chkSVM') && ~isempty(app.ui.chkSVM)
            app.ui.chkSVM.Enable = onOff(isEnabled);
        end
        if isfield(app.ui, 'chkRF') && ~isempty(app.ui.chkRF)
            app.ui.chkRF.Enable = onOff(isEnabled);
        end
        if isfield(app.ui, 'chkXGB') && ~isempty(app.ui.chkXGB)
            app.ui.chkXGB.Enable = onOff(isEnabled);
        end
    end

    function s = onOff(tf)
        if tf
            s = 'on';
        else
            s = 'off';
        end
    end

    function updateBundleStatus()
        [ready, statusText, color] = evaluateBundleStatus(app.model);
        app.bundleReady = ready;
        app.ui.lblBundle.Text = statusText;
        app.ui.lblBundle.FontColor = color;

        if ready
            setAlgorithmCheckboxesEnabled(true);
        else
            setAlgorithmCheckboxesEnabled(false);
        end
    end

    function updateOverallStatus()
        try
            if ~isfield(app, 'ui') || ~isfield(app.ui, 'lblOverall') || isempty(app.ui.lblOverall) || ~isvalid(app.ui.lblOverall)
                return;
            end

            metrics = [];
            if app.activeModelIdx >= 1 && app.activeModelIdx <= numel(app.modelSets)
                metrics = app.modelSets(app.activeModelIdx).metrics;
            end

            if istable(metrics) && ~isempty(metrics) && any(strcmp('Sensitivity', metrics.Properties.VariableNames))
                sens = metrics{:,'Sensitivity'};
                sens = double(sens(:));
                [bestSens, bestIdx] = max(sens);

                algName = "Algorithm " + bestIdx;
                if ~isempty(metrics.Properties.RowNames)
                    try
                        algName = string(metrics.Properties.RowNames{bestIdx});
                    catch
                    end
                end

                app.ui.lblOverall.Text = "Overall best (test Sensitivity): " + algName + " (" + sprintf('%.1f%%', bestSens * 100) + ")";
                app.ui.lblOverall.Tooltip = 'Loaded test metrics from results.mat.';
                app.ui.lblOverall.FontColor = [0.25 0.25 0.25];
            else
                app.ui.lblOverall.Text = 'Overall: (no test metrics)';
                app.ui.lblOverall.Tooltip = 'Place results.mat next to the model file to show test metrics.';
                app.ui.lblOverall.FontColor = [0.45 0.45 0.45];
            end
        catch
        end
    end

    %% ======================== MODEL LOADING ========================
    function loadDefaultModels(modelMatPath)
        if nargin >= 1 && ~isempty(modelMatPath)
            if isfile(modelMatPath)
                addModelFile(modelMatPath, true);
                addModelsFromDir(fileparts(modelMatPath), modelMatPath);
            else
                logLine("Model file not found: " + string(modelMatPath));
            end
            return;
        end

        loadedAny = false;
        loadedAny = addModelsFromDir(fullfile(pwd, 'Training_results'), '') || loadedAny;
        loadedAny = addModelsFromDir(pwd, '') || loadedAny;

        if ~loadedAny
            p = defaultModelPath();
            if isfile(p)
                addModelFile(p, true);
            end
        end
    end

    function loadedAny = addModelsFromDir(dirPath, skipPath)
        loadedAny = false;
        if isempty(dirPath) || ~isfolder(dirPath)
            return;
        end
        d = dir(fullfile(dirPath, 'models*.mat'));
        for i = 1:numel(d)
            fullPath = fullfile(dirPath, d(i).name);
            if ~isempty(skipPath) && strcmp(fullPath, skipPath)
                continue;
            end
            selectAfter = (app.activeModelIdx == 0) && ~loadedAny;
            try
                addModelFile(fullPath, selectAfter);
                loadedAny = true;
            catch me
                logLine("Failed to load model file: " + string(me.message));
            end
        end
    end

    %% ======================== RESULTS TAB ========================
    function loadResultsFile(resultsPath)
        resultsPath = normalizeModelPath(resultsPath);
        s = load(resultsPath);
        if ~isfield(s, 'results')
            error('No "results" struct found in %s.', resultsPath);
        end
        app.results = s.results;
        app.resultsPath = resultsPath;
        refreshResultsTab();
        updateResultsDropdownSelection(app.resultsPath);
        logLine("Loaded results file: " + string(resultsPath));
    end

    function loadDefaultResults()
        app.resultsFiles = collectResultsFiles();
        app.resultsDisplayNames = buildResultsDisplayNames(app.resultsFiles);
        refreshResultsDropdown();

        if isempty(app.results)
            if ~isempty(app.resultsFiles)
                loadResultsFile(app.resultsFiles(1));
            else
                refreshResultsTab();
            end
        else
            updateResultsDropdownSelection(app.resultsPath);
        end
    end

    function files = collectResultsFiles()
        files = strings(0, 1);
        files = [files; listResultsFiles(fullfile(pwd, 'Training_results'))];
        files = [files; listResultsFiles(pwd)];
        if ~isempty(files)
            files = unique(files, 'stable');
        end
    end

    function files = listResultsFiles(dirPath)
        files = strings(0, 1);
        if isempty(dirPath) || ~isfolder(dirPath)
            return;
        end
        d = dir(fullfile(dirPath, 'results*.mat'));
        if isempty(d)
            return;
        end
        names = sort(string({d.name}));
        files = strings(numel(names), 1);
        for i = 1:numel(names)
            files(i) = string(fullfile(dirPath, names(i)));
        end
    end

    function names = buildResultsDisplayNames(files)
        n = numel(files);
        names = strings(n, 1);
        if n == 0
            return;
        end
        bases = strings(n, 1);
        parents = strings(n, 1);
        for i = 1:n
            [p, base, ~] = fileparts(files(i));
            bases(i) = string(base);
            parents(i) = string(getLastPathPart(p));
        end
        names = bases;
        [uBases, ~, idx] = unique(bases);
        for i = 1:numel(uBases)
            mask = idx == i;
            if nnz(mask) > 1
                names(mask) = bases(mask) + " (" + parents(mask) + ")";
            end
        end
    end

    function name = getLastPathPart(pth)
        if isempty(pth)
            name = '';
            return;
        end
        [~, name] = fileparts(pth);
        if isempty(name)
            name = char(pth);
        end
    end

    function refreshResultsDropdown()
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'resultsDropdown') || isempty(app.ui.resultsDropdown)
            return;
        end
        wasUpdating = app.uiUpdating;
        app.uiUpdating = true;
        if isempty(app.resultsFiles)
            app.ui.resultsDropdown.Items = {'(none)'};
            app.ui.resultsDropdown.ItemsData = 0;
            app.ui.resultsDropdown.Value = 0;
            app.ui.resultsDropdown.Enable = 'off';
            app.uiUpdating = wasUpdating;
            return;
        end

        currentValue = app.ui.resultsDropdown.Value;
        n = numel(app.resultsFiles);
        labels = "Results " + (1:n) + ": " + app.resultsDisplayNames(:)';
        app.ui.resultsDropdown.Items = cellstr(labels);
        app.ui.resultsDropdown.ItemsData = 1:n;
        if isempty(currentValue) || ~isnumeric(currentValue) || currentValue < 1 || currentValue > n
            app.ui.resultsDropdown.Value = 1;
        else
            app.ui.resultsDropdown.Value = currentValue;
        end
        app.ui.resultsDropdown.Enable = 'on';
        app.uiUpdating = wasUpdating;
    end

    function updateResultsDropdownSelection(resultsPath)
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'resultsDropdown') || isempty(app.ui.resultsDropdown)
            return;
        end
        wasUpdating = app.uiUpdating;
        app.uiUpdating = true;
        if isempty(resultsPath)
            if ~isempty(app.resultsFiles)
                refreshResultsDropdown();
            else
                app.ui.resultsDropdown.Items = {'(none)'};
                app.ui.resultsDropdown.ItemsData = 0;
                app.ui.resultsDropdown.Value = 0;
                app.ui.resultsDropdown.Enable = 'off';
            end
            app.uiUpdating = wasUpdating;
            return;
        end

        idx = find(app.resultsFiles == string(resultsPath), 1);
        if isempty(idx)
            app.resultsFiles(end+1) = string(resultsPath);
            app.resultsDisplayNames = buildResultsDisplayNames(app.resultsFiles);
            refreshResultsDropdown();
            idx = find(app.resultsFiles == string(resultsPath), 1);
        end
        if ~isempty(idx)
            app.ui.resultsDropdown.Value = idx;
        end
        app.uiUpdating = wasUpdating;
    end

    %% ======================== GRAPH TAB ========================
    function refreshGraphTab()
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'graphPanel') || isempty(app.ui.graphPanel)
            return;
        end

        delete(app.ui.graphPanel.Children);
        try
            app.ui.graphPanel.Scrollable = 'on';
        catch
        end

        imagePaths = listGraphImages();
        if isempty(imagePaths)
            uilabel(app.ui.graphPanel, 'Text', 'No images found in Training_results.', ...
                'HorizontalAlignment', 'center');
            return;
        end

        drawnow;
        n = numel(imagePaths);
        cols = 2;
        rows = ceil(n / cols);
        tileHeight = 260;
        rowSpacing = 10;
        colSpacing = 10;
        padLeft = 8;
        padRight = 8;
        padTop = 8;
        padBottom = 8;

        panelPos = app.ui.graphPanel.Position;
        panelWidth = panelPos(3);
        panelHeight = panelPos(4);
        try
            inner = app.ui.graphPanel.InnerPosition;
            panelWidth = inner(3);
            panelHeight = inner(4);
        catch
        end
        panelWidth = max(panelWidth, 1);
        panelHeight = max(panelHeight, 1);

        tileWidth = (panelWidth - padLeft - padRight - (cols - 1) * colSpacing) / cols;
        tileWidth = max(tileWidth, 1);

        contentHeight = rows * tileHeight + (rows - 1) * rowSpacing + padTop + padBottom;
        contentHeight = max(contentHeight, panelHeight);

        contentPanel = uipanel(app.ui.graphPanel, 'BorderType', 'none');
        contentPanel.Units = 'pixels';
        contentPanel.Position = [0 0 panelWidth contentHeight];

        for i = 1:n
            row = ceil(i / cols);
            col = mod(i - 1, cols) + 1;
            imgPath = char(imagePaths(i));
            [~, base, ext] = fileparts(imgPath);

            x = padLeft + (col - 1) * (tileWidth + colSpacing);
            y = contentHeight - padTop - row * tileHeight - (row - 1) * rowSpacing;
            tile = uipanel(contentPanel, 'Title', char(string(base) + string(ext)));
            tile.Units = 'pixels';
            tile.Position = [x y tileWidth tileHeight];

            ax = uiaxes(tile);
            ax.Units = 'normalized';
            ax.Position = [0.02 0.05 0.96 0.9];
            ax.XTick = [];
            ax.YTick = [];

            try
                I = imread(imgPath);
                imshow(I, 'Parent', ax);
            catch
                title(ax, 'Load error');
            end
        end
    end

    function imagePaths = listGraphImages()
        imagePaths = strings(0, 1);
        folder = fullfile(app.projectRoot, 'Training_Results');
        if ~isfolder(folder)
            return;
        end
        patterns = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'};
        files = strings(0, 1);
        for i = 1:numel(patterns)
            d = dir(fullfile(folder, patterns{i}));
            if isempty(d)
                continue;
            end
            names = sort(string({d.name}));
            files = [files; names(:)]; %#ok<AGROW>
        end
        if isempty(files)
            return;
        end
        files = unique(files, 'stable');
        imagePaths = fullfile(folder, files);
    end

    %% ======================== TRAINING LOG TAB ========================
    function refreshTrainingLogTab()
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'trainingLogText') || isempty(app.ui.trainingLogText)
            return;
        end
        logPath = fullfile(app.projectRoot, 'Training_Results', 'training_log.txt');
        setTextAreaFromFile(app.ui.trainingLogText, logPath, 'Training log file not found.');
    end

    function setTextAreaFromFile(target, path, missingMessage)
        if isempty(target) || ~isvalid(target)
            return;
        end
        if ~isfile(path)
            target.Value = {missingMessage; char("Path: " + string(path))};
            return;
        end
        try
            lines = readlines(path, "EmptyLineRule", "preserve");
        catch
            try
                lines = splitlines(string(fileread(path)));
            catch
                lines = "Failed to read file: " + string(path);
            end
        end
        if isempty(lines)
            lines = "";
        end
        if isstring(lines)
            target.Value = cellstr(lines);
        else
            target.Value = lines;
        end
    end

    function syncResultsTabFromModel()
        if isempty(app.modelSets) || app.activeModelIdx < 1 || app.activeModelIdx > numel(app.modelSets)
            if isempty(app.results)
                refreshResultsTab();
            end
            return;
        end
        entry = app.modelSets(app.activeModelIdx);
        if ~isempty(entry.results)
            app.results = entry.results;
            app.resultsPath = entry.resultsPath;
            updateResultsDropdownSelection(app.resultsPath);
            refreshResultsTab();
        elseif isempty(app.results)
            refreshResultsTab();
        end
    end

    function refreshResultsTab()
        if ~isfield(app, 'ui') || ~isfield(app.ui, 'resultsDropdown') || isempty(app.ui.resultsDropdown)
            return;
        end

        if isempty(app.results)
            updateResultsDropdownSelection('');
            clearTable(app.ui.tblTestResults);
            clearTable(app.ui.tblMetrics);
            clearTable(app.ui.tblConfSVM);
            clearTable(app.ui.tblConfRF);
            clearTable(app.ui.tblConfXGB);
            clearTable(app.ui.tblCfg);
            clearTable(app.ui.tblTiming);
            clearRocPlot(app.ui.axRocSVM, 'SVM');
            clearRocPlot(app.ui.axRocRF, 'RF');
            clearRocPlot(app.ui.axRocXGB, 'XGB');
            return;
        end

        updateResultsDropdownSelection(app.resultsPath);

        [data, colNames, rowNames] = buildTestResultsTable(app.results);
        app.ui.tblTestResults.Data = data;
        app.ui.tblTestResults.ColumnName = colNames;
        app.ui.tblTestResults.RowName = rowNames;
        try
            app.ui.tblTestResults.ColumnEditable = false(1, size(data, 2));
        catch
        end

        [sumData, sumCols, sumRows] = buildMetricsSummary(app.results);
        app.ui.tblMetrics.Data = sumData;
        app.ui.tblMetrics.ColumnName = sumCols;
        app.ui.tblMetrics.RowName = sumRows;
        try
            app.ui.tblMetrics.ColumnEditable = false(1, size(sumData, 2));
        catch
        end

        updateConfusionTables(app.results);
        updateRocPlots(app.results);
        updateCfgTimingTables(app.results);
    end

    function clearTable(tbl)
        if isempty(tbl) || ~isvalid(tbl)
            return;
        end
        tbl.Data = cell(0, 0);
        tbl.ColumnName = {};
        tbl.RowName = {};
    end

    function clearRocPlot(ax, titleText)
        if isempty(ax) || ~isvalid(ax)
            return;
        end
        cla(ax);
        title(ax, titleText);
        ax.XLim = [0 1];
        ax.YLim = [0 1];
        grid(ax, 'on');
    end

    function [data, colNames, rowNames] = buildTestResultsTable(results)
        data = cell(0, 0);
        colNames = {};
        rowNames = {};

        if isfield(results, 'testResults') && istable(results.testResults)
            tbl = results.testResults;
            data = table2cell(tbl);
            colNames = tbl.Properties.VariableNames;
            rowNames = tbl.Properties.RowNames;
            if isempty(rowNames)
                rowNames = {};
            end
            return;
        end

        if isfield(results, 'comparisonTable') && istable(results.comparisonTable)
            tbl = results.comparisonTable;
            data = table2cell(tbl);
            colNames = tbl.Properties.VariableNames;
            rowNames = tbl.Properties.RowNames;
            if isempty(rowNames)
                rowNames = {};
            end
        end
    end

    function [data, colNames, rowNames] = buildMetricsSummary(results)
        data = cell(0, 0);
        colNames = {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'AUC'};
        rowNames = {};

        algFields = {'res_SVM', 'res_RF', 'res_XGB'};
        algLabels = {'SVM', 'RF', 'XGB'};
        for i = 1:numel(algFields)
            if ~isfield(results, algFields{i})
                continue;
            end
            res = results.(algFields{i});
            row = cell(1, numel(colNames));
            for j = 1:numel(colNames)
                row{j} = fmtMetric(getMetricValue(res, colNames{j}));
            end
            data(end+1, :) = row; %#ok<AGROW>
            rowNames{end+1, 1} = algLabels{i}; %#ok<AGROW>
        end
    end

    function v = getMetricValue(res, name)
        v = NaN;
        if ~isstruct(res) || ~isfield(res, name)
            return;
        end
        tmp = res.(name);
        if isnumeric(tmp) && ~isempty(tmp)
            v = tmp(1);
        end
    end

    function s = fmtMetric(v)
        if ~isnumeric(v) || isempty(v) || ~isfinite(v)
            s = '-';
            return;
        end
        s = sprintf('%.3f', v);
    end

    function updateConfusionTables(results)
        labels = getClassLabels(results);
        updateConfusionTable(app.ui.tblConfSVM, getConfusionMatrix(results, 'res_SVM'), labels);
        updateConfusionTable(app.ui.tblConfRF, getConfusionMatrix(results, 'res_RF'), labels);
        updateConfusionTable(app.ui.tblConfXGB, getConfusionMatrix(results, 'res_XGB'), labels);
    end

    function labels = getClassLabels(results)
        labels = strings(0, 1);
        if isfield(results, 'orderStr') && ~isempty(results.orderStr)
            try
                labels = string(results.orderStr);
            catch
            end
        end
        if numel(labels) < 2 && isfield(results, 'cfg')
            if isfield(results.cfg, 'negClassName') && isfield(results.cfg, 'posClassName')
                labels = [string(results.cfg.negClassName), string(results.cfg.posClassName)];
            end
        end
        if numel(labels) < 2
            labels = ["benign", "malignant"];
        end
        labels = labels(:)';
    end

    function cm = getConfusionMatrix(results, fieldName)
        cm = [];
        if isfield(results, fieldName)
            res = results.(fieldName);
            if isstruct(res) && isfield(res, 'CM')
                cm = res.CM;
            end
        end
    end

    function updateConfusionTable(tbl, cm, labels)
        if isempty(tbl) || ~isvalid(tbl)
            return;
        end
        if isempty(cm)
            clearTable(tbl);
            return;
        end
        cm = double(cm);
        tbl.Data = cm;
        if numel(labels) == size(cm, 2)
            tbl.ColumnName = cellstr(labels);
        else
            tbl.ColumnName = {};
        end
        if numel(labels) == size(cm, 1)
            tbl.RowName = cellstr(labels);
        else
            tbl.RowName = {};
        end
        try
            tbl.ColumnEditable = false(1, size(cm, 2));
        catch
        end
    end

    function updateRocPlots(results)
        plotRoc(app.ui.axRocSVM, getRoc(results, 'res_SVM'), 'SVM');
        plotRoc(app.ui.axRocRF, getRoc(results, 'res_RF'), 'RF');
        plotRoc(app.ui.axRocXGB, getRoc(results, 'res_XGB'), 'XGB');
    end

    function roc = getRoc(results, fieldName)
        roc = [];
        if isfield(results, fieldName)
            res = results.(fieldName);
            if isstruct(res) && isfield(res, 'ROC')
                roc = res.ROC;
            end
        end
    end

    function plotRoc(ax, roc, labelText)
        if isempty(ax) || ~isvalid(ax)
            return;
        end
        cla(ax);
        if isempty(roc) || ~isstruct(roc) || ~isfield(roc, 'FPR') || ~isfield(roc, 'TPR')
            title(ax, labelText);
            ax.XLim = [0 1];
            ax.YLim = [0 1];
            grid(ax, 'on');
            return;
        end
        fpr = double(roc.FPR(:));
        tpr = double(roc.TPR(:));
        plot(ax, fpr, tpr, 'LineWidth', 1.5);
        hold(ax, 'on');
        plot(ax, [0 1], [0 1], '--', 'Color', [0.7 0.7 0.7]);
        hold(ax, 'off');
        ax.XLim = [0 1];
        ax.YLim = [0 1];
        grid(ax, 'on');
        xlabel(ax, 'FPR');
        ylabel(ax, 'TPR');
        if isfield(roc, 'AUC') && isnumeric(roc.AUC) && isfinite(roc.AUC)
            title(ax, sprintf('%s (AUC=%.3f)', labelText, roc.AUC));
        else
            title(ax, labelText);
        end
    end

    function updateCfgTimingTables(results)
        if isfield(results, 'cfg') && isstruct(results.cfg)
            [cfgData, cfgCols] = structToKeyValueTable(results.cfg, false);
            app.ui.tblCfg.Data = cfgData;
            app.ui.tblCfg.ColumnName = cfgCols;
            app.ui.tblCfg.RowName = {};
            try
                app.ui.tblCfg.ColumnEditable = false(1, size(cfgData, 2));
            catch
            end
        else
            clearTable(app.ui.tblCfg);
        end

        if isfield(results, 'timing') && isstruct(results.timing)
            [timingData, timingCols] = structToKeyValueTable(results.timing, true);
            app.ui.tblTiming.Data = timingData;
            app.ui.tblTiming.ColumnName = timingCols;
            app.ui.tblTiming.RowName = {};
            try
                app.ui.tblTiming.ColumnEditable = false(1, size(timingData, 2));
            catch
            end
        else
            clearTable(app.ui.tblTiming);
        end
    end

    function [data, colNames] = structToKeyValueTable(s, isTiming)
        data = cell(0, 2);
        colNames = {'Field', 'Value'};
        if ~isstruct(s)
            return;
        end
        fields = fieldnames(s);
        fields = sort(fields);
        data = cell(numel(fields), 2);
        for i = 1:numel(fields)
            data{i, 1} = fields{i};
            if isTiming
                data{i, 2} = timingValueToString(s.(fields{i}));
            else
                data{i, 2} = valueToString(s.(fields{i}), 0);
            end
        end
    end

    function s = timingValueToString(v)
        if isnumeric(v) && isscalar(v) && isfinite(v)
            s = sprintf('%.3f s', v);
            return;
        end
        s = valueToString(v, 0);
    end

    function s = valueToString(v, depth)
        if nargin < 2
            depth = 0;
        end
        if isstring(v)
            if numel(v) == 1
                s = char(v);
            else
                s = char(strjoin(v(:)', ", "));
            end
            return;
        end
        if ischar(v)
            s = v;
            return;
        end
        if isnumeric(v) || islogical(v)
            if isempty(v)
                s = '[]';
            elseif isscalar(v)
                s = num2str(v);
            else
                s = mat2str(v);
            end
            return;
        end
        if iscell(v)
            try
                s = char(strjoin(string(v(:)'), ", "));
                return;
            catch
            end
            s = ['cell ' mat2str(size(v))];
            return;
        end
        if isstruct(v)
            f = fieldnames(v);
            if isempty(f)
                s = 'struct()';
                return;
            end
            if depth >= 1
                s = ['struct fields: ' strjoin(f, ', ')];
                return;
            end
            parts = strings(0, 1);
            for i = 1:numel(f)
                parts(end+1) = string(f{i}) + "=" + string(valueToString(v.(f{i}), depth + 1));
            end
            s = char("struct(" + strjoin(parts, ", ") + ")");
            return;
        end
        s = class(v);
    end

    function exportResultsCsv(targetPath)
        [outDir, baseName, ~] = fileparts(targetPath);
        if isempty(outDir)
            outDir = pwd;
        end
        if isempty(baseName)
            baseName = 'results_export';
        end

        exported = strings(0, 1);
        testTbl = getTestResultsTable(app.results);
        if ~isempty(testTbl)
            p = fullfile(outDir, baseName + "_testResults.csv");
            writeTableCsv(testTbl, p);
            exported(end+1) = string(p);
        end

        metricsTbl = getMetricsSummaryTable(app.results);
        if ~isempty(metricsTbl)
            p = fullfile(outDir, baseName + "_metricsSummary.csv");
            writeTableCsv(metricsTbl, p);
            exported(end+1) = string(p);
        end

        cfgTbl = getCfgTable(app.results);
        if ~isempty(cfgTbl)
            p = fullfile(outDir, baseName + "_cfg.csv");
            writeTableCsv(cfgTbl, p);
            exported(end+1) = string(p);
        end

        timingTbl = getTimingTable(app.results);
        if ~isempty(timingTbl)
            p = fullfile(outDir, baseName + "_timing.csv");
            writeTableCsv(timingTbl, p);
            exported(end+1) = string(p);
        end

        svmCm = getConfusionTable(app.results, 'res_SVM');
        if ~isempty(svmCm)
            p = fullfile(outDir, baseName + "_confusion_SVM.csv");
            writeTableCsv(svmCm, p);
            exported(end+1) = string(p);
        end

        rfCm = getConfusionTable(app.results, 'res_RF');
        if ~isempty(rfCm)
            p = fullfile(outDir, baseName + "_confusion_RF.csv");
            writeTableCsv(rfCm, p);
            exported(end+1) = string(p);
        end

        xgbCm = getConfusionTable(app.results, 'res_XGB');
        if ~isempty(xgbCm)
            p = fullfile(outDir, baseName + "_confusion_XGB.csv");
            writeTableCsv(xgbCm, p);
            exported(end+1) = string(p);
        end

        svmRoc = getRocTable(app.results, 'res_SVM');
        if ~isempty(svmRoc)
            p = fullfile(outDir, baseName + "_roc_SVM.csv");
            writeTableCsv(svmRoc, p);
            exported(end+1) = string(p);
        end

        rfRoc = getRocTable(app.results, 'res_RF');
        if ~isempty(rfRoc)
            p = fullfile(outDir, baseName + "_roc_RF.csv");
            writeTableCsv(rfRoc, p);
            exported(end+1) = string(p);
        end

        xgbRoc = getRocTable(app.results, 'res_XGB');
        if ~isempty(xgbRoc)
            p = fullfile(outDir, baseName + "_roc_XGB.csv");
            writeTableCsv(xgbRoc, p);
            exported(end+1) = string(p);
        end

        if isempty(exported)
            logLine('No results tables to export.');
            return;
        end
        for i = 1:numel(exported)
            logLine("Exported: " + exported(i));
        end
    end

    function tbl = getTestResultsTable(results)
        tbl = table();
        if isfield(results, 'testResults') && istable(results.testResults)
            tbl = results.testResults;
        elseif isfield(results, 'comparisonTable') && istable(results.comparisonTable)
            tbl = results.comparisonTable;
        end
    end

    function tbl = getMetricsSummaryTable(results)
        tbl = table();
        [data, colNames, rowNames] = buildMetricsSummary(results);
        if isempty(data)
            return;
        end
        tbl = cell2table(data, 'VariableNames', colNames);
        if ~isempty(rowNames)
            tbl.Properties.RowNames = rowNames;
        end
    end

    function tbl = getConfusionTable(results, fieldName)
        tbl = table();
        cm = getConfusionMatrix(results, fieldName);
        if isempty(cm)
            return;
        end
        cm = double(cm);
        labels = getClassLabels(results);
        if numel(labels) == size(cm, 2)
            varNames = cellstr(labels);
        else
            varNames = cellstr("C" + string(1:size(cm, 2)));
        end
        tbl = array2table(cm, 'VariableNames', varNames);
        if numel(labels) == size(cm, 1)
            tbl.Properties.RowNames = cellstr(labels);
        end
    end

    function tbl = getRocTable(results, fieldName)
        tbl = table();
        roc = getRoc(results, fieldName);
        if isempty(roc) || ~isstruct(roc) || ~isfield(roc, 'FPR') || ~isfield(roc, 'TPR')
            return;
        end
        fpr = double(roc.FPR(:));
        tpr = double(roc.TPR(:));
        auc = NaN;
        if isfield(roc, 'AUC') && isnumeric(roc.AUC) && isfinite(roc.AUC)
            auc = roc.AUC;
        end
        aucCol = repmat(auc, numel(fpr), 1);
        tbl = table(fpr, tpr, aucCol, 'VariableNames', {'FPR', 'TPR', 'AUC'});
    end

    function tbl = getCfgTable(results)
        tbl = table();
        if isfield(results, 'cfg') && isstruct(results.cfg)
            [data, colNames] = structToKeyValueTable(results.cfg, false);
            if ~isempty(data)
                tbl = cell2table(data, 'VariableNames', colNames);
            end
        end
    end

    function tbl = getTimingTable(results)
        tbl = table();
        if isfield(results, 'timing') && isstruct(results.timing)
            [data, colNames] = structToKeyValueTable(results.timing, true);
            if ~isempty(data)
                tbl = cell2table(data, 'VariableNames', colNames);
            end
        end
    end

    function writeTableCsv(tbl, path)
        if isempty(tbl) || (height(tbl) == 0 && width(tbl) == 0)
            return;
        end
        if isempty(tbl.Properties.RowNames)
            writetable(tbl, path);
        else
            writetable(tbl, path, 'WriteRowNames', true);
        end
    end

    function [labelStr, confStr, classNames, scoreProb] = predictSingle(I, modelKey)
        cfg = app.model.cfg;
        feat0 = extractFeaturesFromImage(I, cfg);
        [featSelected, warnMsg] = applyFeatureSelectionForInference(feat0, app.model);
        if strlength(warnMsg) > 0
            logLine(warnMsg);
        end

        [featureRow, branchAInput, branchBInput] = prepareBranchInputs(featSelected, app.model);
        key = normalizeAlgorithmKey(modelKey);

        switch key
            case 'SVM'
                if cfg.useLDA_for_SVM
                    if isempty(branchAInput)
                        error('SVM requires LDA (missing ldaModel or zParams).');
                    end
                    Xrow = branchAInput;
                else
                    if isempty(branchBInput)
                        error('SVM requires normalization params.');
                    end
                    Xrow = branchBInput;
                end
                [predLabel, rawScores, classNames] = predictWithClassifier(app.model.trained.models.SVM, Xrow);
            case 'RF'
                [predLabel, rawScores, classNames] = predictWithRandomForest(app.model.trained.models.RF, branchBInput);
            case 'XGB'
                [predLabel, rawScores, classNames] = predictWithClassifier(app.model.trained.models.XGB, branchBInput);
            otherwise
                error('Unknown model: %s', modelKey);
        end

        [scoreProb, conf] = normalizeScores(rawScores);
        labelStr = string(predLabel);
        confStr = sprintf('%.3f', conf);
    end

    function [featureRow, branchAInput, branchBInput] = prepareBranchInputs(featSelected, model)
        featureRow = [];
        branchAInput = [];
        branchBInput = [];

        if isempty(featSelected)
            return;
        end

        featureRow = double(featSelected(:)');

        if ~isfield(model, 'trained') || ~isfield(model.trained, 'branchB') || ~isfield(model.trained.branchB, 'zParams')
            return;
        end

        zParams = model.trained.branchB.zParams;
        mu = zParams.mu;
        sig = zParams.sig;
        if numel(featureRow) ~= numel(mu)
            error('Feature dimension mismatch: got %d, expected %d.', numel(featureRow), numel(mu));
        end
        branchBInput = (featureRow - mu) ./ (sig + eps);

        if isfield(model, 'cfg') && isfield(model.cfg, 'useLDA_for_SVM') && model.cfg.useLDA_for_SVM
            if isfield(model.trained, 'branchA') && isfield(model.trained.branchA, 'ldaModel') && ~isempty(model.trained.branchA.ldaModel)
                [~, sc] = predict(model.trained.branchA.ldaModel, branchBInput);
                posCol = min(2, size(sc, 2));
                branchAInput = sc(:, posCol);
            end
        end
    end

    function algs = getSelectedAlgorithms()
        algs = strings(0, 1);
        if isfield(app.ui, 'chkSVM') && app.ui.chkSVM.Value
            algs(end+1) = "SVM";
        end
        if isfield(app.ui, 'chkRF') && app.ui.chkRF.Value
            algs(end+1) = "Random Forest";
        end
        if isfield(app.ui, 'chkXGB') && app.ui.chkXGB.Value
            algs(end+1) = "XGBoost";
        end
    end

    function results = predictAlgorithmsList(I, algs)
        if isempty(algs)
            results = struct([]);
            return;
        end
        first = predictAlgorithmSafe(I, algs(1));
        results = repmat(first, 1, numel(algs));
        for i = 2:numel(algs)
            results(i) = predictAlgorithmSafe(I, algs(i));
        end
    end

    function results = predictAllAlgorithms(I)
        algs = ["SVM", "Random Forest", "XGBoost"];
        results = predictAlgorithmsList(I, algs);
    end

    function r = predictAlgorithmSafe(I, modelKey)
        r = struct();
        r.algorithm = string(modelKey);
        r.ok = false;
        r.label = "";
        r.conf = NaN;
        r.classNames = strings(0, 1);
        r.scores = [];
        r.benign = NaN;
        r.malignant = NaN;
        r.message = "";

        try
            [labelStr, confStr, classNames, scoreProb] = predictSingle(I, modelKey);
        catch me
            r.message = string(me.message);
            return;
        end

        r.ok = true;
        r.label = string(labelStr);
        r.conf = str2double(string(confStr));
        r.classNames = string(classNames);
        r.scores = scoreProb;
        [r.benign, r.malignant] = extractBenignMalignant(r.classNames, r.scores, app.model.orderStr);
    end

    function best = showCompareResults(results)
        best = struct('ok', false);

        if ~isstruct(results) || isempty(results)
            return;
        end

        % Decide "best" by confidence (max over normalized scores)
        bestIdx = 0;
        bestConf = -inf;
        for i = 1:numel(results)
            if results(i).ok && isfinite(results(i).conf) && results(i).conf > bestConf
                bestConf = results(i).conf;
                bestIdx = i;
            end
        end
        if bestIdx == 0
            for i = 1:numel(results)
                if results(i).ok
                    bestIdx = i;
                    break;
                end
            end
        end

        % Update compare table (if present)
        if isfield(app.ui, 'tblCompare') && ~isempty(app.ui.tblCompare)
            tableData = cell(numel(results), 5);
            for i = 1:numel(results)
                algName = results(i).algorithm;
                if numel(results) > 1 && i == bestIdx
                    algName = algName + " *";
                end
                tableData{i, 1} = char(algName);

                if results(i).ok
                    tableData{i, 2} = char(results(i).label);
                    tableData{i, 3} = fmtScore(results(i).conf);
                    tableData{i, 4} = fmtScore(results(i).benign);
                    tableData{i, 5} = fmtScore(results(i).malignant);
                else
                    tableData{i, 2} = 'Error';
                    tableData{i, 3} = '-';
                    tableData{i, 4} = '-';
                    tableData{i, 5} = '-';
                end
            end
            app.ui.tblCompare.Data = tableData;
        end

        if bestIdx > 0
            best = results(bestIdx); %#ok<NASGU>
        end
        showCompareBars(results, bestIdx);
    end

    function [benignScore, malignantScore] = extractBenignMalignant(classNames, scores, orderStr)
        benignScore = NaN;
        malignantScore = NaN;
        if isempty(classNames) || isempty(scores) || numel(scores) < 2
            return;
        end

        names = lower(string(classNames(:)'));
        scores = double(scores(:)');
        if isempty(orderStr) || numel(orderStr) < 2
            orderStr = ["benign", "malignant"];
        end
        idxBenign = find(names == lower(string(orderStr(1))), 1);
        idxMalig = find(names == lower(string(orderStr(2))), 1);
        if isempty(idxBenign) || isempty(idxMalig)
            return;
        end

        benignScore = scores(idxBenign);
        malignantScore = scores(idxMalig);
    end

    function s = fmtScore(v)
        if ~isfinite(v)
            s = '-';
            return;
        end
        s = sprintf('%.3f', v);
    end

    %% ======================== DISPLAY ========================
    function showImage(I)
        cla(app.ui.axImage);
        try
            if ndims(I) == 2 || (ndims(I) == 3 && size(I, 3) == 1)
                imshow(I, [], 'Parent', app.ui.axImage);
            else
                Irgb = toRgb3(I);
                imshow(Irgb, 'Parent', app.ui.axImage);
            end
        catch me
            logLine("Preview error: " + string(me.message));
            try
                if ndims(I) >= 2
                    imshow(I(:, :, 1), [], 'Parent', app.ui.axImage);
                end
            catch
            end
        end
    end

    function showCompareBars(results, bestIdx)
        if ~isfield(app.ui, 'axCompare') || isempty(app.ui.axCompare) || ~isvalid(app.ui.axCompare)
            return;
        end

        cla(app.ui.axCompare);
        if ~isstruct(results) || isempty(results)
            title(app.ui.axCompare, 'Algorithm Comparison');
            app.ui.axCompare.YLim = [0 1];
            grid(app.ui.axCompare, 'on');
            return;
        end

        n = numel(results);
        scores = nan(n, 2);
        labels = strings(n, 1);
        for i = 1:n
            labels(i) = results(i).algorithm;
            if nargin >= 2 && bestIdx > 0 && numel(results) > 1 && i == bestIdx
                labels(i) = labels(i) + " *";
            end
            if isfield(results(i), 'ok') && results(i).ok
                scores(i, 1) = results(i).benign;
                scores(i, 2) = results(i).malignant;
            end
        end

        if all(~isfinite(scores(:)))
            title(app.ui.axCompare, 'Algorithm Comparison');
            app.ui.axCompare.YLim = [0 1];
            grid(app.ui.axCompare, 'on');
            return;
        end

        scoresPlot = scores;
        nPlot = n;
        if n == 1
            scoresPlot = [scores; nan(1, size(scores, 2))];
            nPlot = 2;
        end

        b = bar(app.ui.axCompare, scoresPlot, 'grouped');
        if numel(b) >= 1
            b(1).FaceColor = [0.25 0.6 0.9];
        end
        if numel(b) >= 2
            b(2).FaceColor = [0.9 0.4 0.4];
        end
        app.ui.axCompare.XTick = 1:n;
        app.ui.axCompare.XTickLabel = cellstr(labels);
        app.ui.axCompare.XTickLabelRotation = 20;
        app.ui.axCompare.YLim = [0 1];
        if n == 1
            app.ui.axCompare.XLim = [0.5 1.5];
        else
            app.ui.axCompare.XLim = [0.5 nPlot + 0.5];
        end
        ylabel(app.ui.axCompare, 'Normalized score');
        grid(app.ui.axCompare, 'on');
        legend(app.ui.axCompare, {'Benign', 'Malignant'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
        title(app.ui.axCompare, 'Algorithm Comparison');
    end

    function logLine(msg)
        try
            if ~isfield(app, 'ui') || ~isfield(app.ui, 'log') || isempty(app.ui.log) || ~isvalid(app.ui.log)
                return;
            end
            ts = string(datetime('now', 'Format', 'HH:mm:ss'));
            line = ts + "  " + string(msg);

            existing = app.ui.log.Value;
            if isempty(existing)
                app.ui.log.Value = {char(line)};
            elseif iscell(existing)
                existing = existing(:);
                existing{end+1, 1} = char(line);
                app.ui.log.Value = existing;
            elseif isstring(existing)
                existing = existing(:);
                app.ui.log.Value = [existing; line];
            elseif ischar(existing)
                app.ui.log.Value = {existing; char(line)};
            else
                app.ui.log.Value = {char(line)};
            end

            % Keep the last ~200 lines to avoid UI slowdown.
            if iscell(app.ui.log.Value) && numel(app.ui.log.Value) > 200
                app.ui.log.Value = app.ui.log.Value(end-199:end);
            elseif isstring(app.ui.log.Value) && numel(app.ui.log.Value) > 200
                app.ui.log.Value = app.ui.log.Value(end-199:end);
            end

            drawnow limitrate;
            try
                if isprop(app.ui.log, 'ScrollTo')
                    app.ui.log.ScrollTo = 'bottom';
                elseif ismethod(app.ui.log, 'scroll')
                    scroll(app.ui.log, 'bottom');
                elseif isprop(app.ui.log, 'Selection')
                    val = app.ui.log.Value;
                    if iscell(val)
                        text = strjoin(string(val), newline);
                    else
                        text = string(val);
                    end
                    n = strlength(text);
                    app.ui.log.Selection = [n n];
                end
            catch
            end
        catch
        end
    end

    %% ======================== MODEL LOAD HELPERS ========================
    function entry = buildModelEntry(pathToMat, trained, results, resultsPath)
        [~, base] = fileparts(pathToMat);
        displayName = makeUniqueModelSetName(string(base));

        entry = struct();
        entry.trained = trained;
        entry.results = results;
        entry.cfg = [];
        entry.fsInfo = [];
        entry.orderStr = [];
        entry.path = char(pathToMat);
        entry.resultsPath = '';
        entry.displayName = char(displayName);
        entry.metrics = [];

        if nargin >= 4 && ~isempty(resultsPath)
            entry.resultsPath = char(resultsPath);
        elseif ~isempty(results) && isfield(results, 'cfg')
            entry.resultsPath = char(pathToMat);
        end

        entry = applyResultsMetadata(entry);
    end

    function entry = applyResultsMetadata(entry)
        if ~isempty(entry.results)
            if isfield(entry.results, 'cfg')
                entry.cfg = entry.results.cfg;
            end
            if isfield(entry.results, 'fsInfo')
                entry.fsInfo = entry.results.fsInfo;
            end
            if isfield(entry.results, 'orderStr')
                entry.orderStr = entry.results.orderStr;
            end
            if isfield(entry.results, 'testResults')
                entry.metrics = entry.results.testResults;
            end
        end

        if isempty(entry.orderStr) && ~isempty(entry.cfg) && isfield(entry.cfg, 'negClassName') && isfield(entry.cfg, 'posClassName')
            entry.orderStr = lower(string([entry.cfg.negClassName, entry.cfg.posClassName]));
        elseif isempty(entry.orderStr)
            entry.orderStr = ["benign", "malignant"];
        end
    end

    function [trained, results, resultsPath] = loadModelBundle(modelMatPath)
        s = load(modelMatPath);
        trained = [];
        results = [];
        resultsPath = '';

        if isfield(s, 'trained')
            trained = s.trained;
        end
        if isfield(s, 'results')
            results = s.results;
            resultsPath = modelMatPath;
        end

        if isempty(results) || ~isfield(results, 'cfg')
            [sRes, resPath] = tryLoadResultsFile(modelMatPath);
            if ~isempty(sRes)
                if isempty(results) && isfield(sRes, 'results')
                    results = sRes.results;
                end
                if isempty(trained) && isfield(sRes, 'trained')
                    trained = sRes.trained;
                end
                if isempty(resultsPath)
                    resultsPath = resPath;
                end
            end
        end
    end

    function [ready, statusText, color] = evaluateBundleStatus(model)
        ready = false;
        statusText = 'Bundle: -';
        color = [0.5 0.5 0.5];

        if isempty(model) || ~isfield(model, 'trained') || isempty(model.trained)
            statusText = 'Bundle: missing trained models';
            color = [0.65 0.2 0.2];
            return;
        end
        if ~isfield(model, 'cfg') || isempty(model.cfg)
            statusText = 'Bundle: missing cfg (attach results.mat)';
            color = [0.65 0.2 0.2];
            return;
        end
        if ~isfield(model.trained, 'featureIndices') || isempty(model.trained.featureIndices)
            statusText = 'Bundle: missing feature indices';
            color = [0.65 0.2 0.2];
            return;
        end
        if ~isfield(model.trained, 'branchB') || ~isfield(model.trained.branchB, 'zParams')
            statusText = 'Bundle: missing normalization params';
            color = [0.65 0.2 0.2];
            return;
        end

        ready = true;
        statusText = 'Bundle: ready';
        color = [0.15 0.5 0.2];

        warn = strings(0, 1);
        if ~isfield(model, 'fsInfo') || ~isfield(model.fsInfo, 'keepMask') || isempty(model.fsInfo.keepMask)
            warn(end+1) = "missing fsInfo.keepMask";
        end
        if isfield(model.cfg, 'useLDA_for_SVM') && model.cfg.useLDA_for_SVM
            if ~isfield(model.trained, 'branchA') || ~isfield(model.trained.branchA, 'ldaModel') || isempty(model.trained.branchA.ldaModel)
                warn(end+1) = "SVM disabled (missing LDA)";
            end
        end

        if ~isempty(warn)
            statusText = "Bundle: ready (" + strjoin(warn, "; ") + ")";
            color = [0.65 0.45 0.1];
        end
    end

    function p = normalizeModelPath(pth)
        p = char(pth);
        if ~isfile(p)
            error('Model file not found: %s', p);
        end
        if ~isAbsolutePath(p)
            p = fullfile(pwd, p);
        end
        p = char(string(p));
    end

    function name = makeUniqueModelSetName(baseName)
        baseName = string(baseName);
        existing = string({app.modelSets.displayName});
        name = baseName;
        k = 2;
        while any(existing == name)
            name = baseName + " (" + k + ")";
            k = k + 1;
        end
    end
end

%% =====================================================================
%% Inference helpers and feature extraction (copied/adapted)
%% =====================================================================

function p = defaultModelPath()
    p = 'models.mat';
    if isfile(p)
        return;
    end

    candidate = fullfile(pwd, 'Training_results', 'models.mat');
    if isfile(candidate)
        p = candidate;
        return;
    end

    d = dir(fullfile(pwd, 'Training_results', 'models*.mat'));
    if ~isempty(d)
        p = fullfile(pwd, 'Training_results', d(1).name);
        return;
    end

    d = dir(fullfile(pwd, 'models*.mat'));
    if ~isempty(d)
        p = fullfile(pwd, d(1).name);
        return;
    end

    p = 'results.mat';
    if ~isfile(p)
        candidate = fullfile(pwd, 'Training_results', 'results.mat');
        if isfile(candidate)
            p = candidate;
        end
    end
end

function p = defaultResultsPath()
    p = 'results.mat';
    if isfile(p)
        return;
    end
    candidate = fullfile(pwd, 'Training_results', 'results.mat');
    if isfile(candidate)
        p = candidate;
        return;
    end
    d = dir(fullfile(pwd, 'Training_results', 'results*.mat'));
    if ~isempty(d)
        p = fullfile(pwd, 'Training_results', d(1).name);
        return;
    end
    d = dir(fullfile(pwd, 'results*.mat'));
    if ~isempty(d)
        p = fullfile(pwd, d(1).name);
        return;
    end
    p = '';
end

function [sRes, resPath] = tryLoadResultsFile(modelMatPath)
    sRes = [];
    resPath = '';
    try
        [p, base] = fileparts(modelMatPath);
        candidates = strings(0, 1);
        if contains(lower(base), "results")
            candidates(end+1) = string(modelMatPath);
        end
        candidates(end+1) = fullfile(p, 'results.mat');
        d = dir(fullfile(p, 'results*.mat'));
        for i = 1:numel(d)
            candidates(end+1) = fullfile(p, d(i).name); %#ok<AGROW>
        end

        for i = 1:numel(candidates)
            candPath = char(candidates(i));
            if isempty(candPath) || ~isfile(candPath)
                continue;
            end
            sTmp = load(candPath);
            if isfield(sTmp, 'results')
                sRes = sTmp;
                resPath = candPath;
                return;
            end
        end
    catch
        sRes = [];
        resPath = '';
    end
end

function [featSelected, warnMsg] = applyFeatureSelectionForInference(featRow, model)
    warnMsg = "";
    featSelected = featRow(:)';

    if isfield(model, 'fsInfo') && isfield(model.fsInfo, 'keepMask') && ~isempty(model.fsInfo.keepMask)
        keepMask = model.fsInfo.keepMask;
        if numel(keepMask) == numel(featSelected)
            featSelected = featSelected(keepMask);
        else
            warnMsg = "fsInfo.keepMask size mismatch; using raw features.";
        end
    else
        warnMsg = "fsInfo.keepMask missing; using raw features.";
    end

    if isfield(model, 'trained') && isfield(model.trained, 'featureIndices') && ~isempty(model.trained.featureIndices)
        idx = model.trained.featureIndices(:);
        if max(idx) > numel(featSelected)
            error('Feature index out of range: max=%d, available=%d.', max(idx), numel(featSelected));
        end
        featSelected = featSelected(idx);
    end
end

function [predLabel, rawScores, classNames] = predictWithClassifier(modelObj, Xrow)
    try
        [predLabel, rawScores] = predict(modelObj, Xrow);
    catch
        predLabel = predict(modelObj, Xrow);
        rawScores = [];
    end

    if isprop(modelObj, 'ClassNames')
        classNames = modelObj.ClassNames;
    else
        classNames = [];
    end
end

function [predLabel, rawScores, classNames] = predictWithRandomForest(rfModel, Xrow)
    [predCell, rawScores] = predict(rfModel, Xrow);
    if iscell(predCell)
        predLabel = string(predCell{1});
    else
        predLabel = string(predCell);
    end
    classNames = rfModel.ClassNames;
end

function [scoreProb, conf] = normalizeScores(rawScores)
    if isempty(rawScores)
        scoreProb = [NaN NaN];
        conf = NaN;
        return;
    end
    s = double(rawScores(1, :));
    if any(~isfinite(s))
        scoreProb = zeros(size(s));
        conf = 0;
        return;
    end
    if all(s >= 0) && abs(sum(s) - 1) < 1e-6
        scoreProb = s;
    else
        scoreProb = softmaxRow(s);
    end
    conf = max(scoreProb);
end

function y = softmaxRow(x)
    x = x(:)';
    x = x - max(x);
    ex = exp(x);
    y = ex ./ (sum(ex) + eps);
end

function key = normalizeAlgorithmKey(name)
    n = lower(strrep(string(name), ' ', ''));
    switch n
        case 'svm'
            key = 'SVM';
        case 'randomforest'
            key = 'RF';
        case 'xgboost'
            key = 'XGB';
        otherwise
            key = char(name);
    end
end

function tf = isAbsolutePath(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep);
    if ~tf && numel(pth) >= 2
        tf = isletter(pth(1)) && pth(2) == ':';
    end
end

function Irgb = toRgb3(I)
    if ndims(I) == 2
        Irgb = repmat(I, [1 1 3]);
        return;
    end
    if size(I, 3) == 1
        Irgb = repmat(I, [1 1 3]);
    elseif size(I, 3) == 2
        Irgb = cat(3, I(:, :, 1), I(:, :, 2), I(:, :, 2));
    else
        Irgb = I(:, :, 1:3);
    end
end

function feat = extractFeaturesFromImage(I, cfg)
    Iproc = advancedPreprocessing(I, cfg);
    mask = basicSegmentationMask(Iproc);
    % Collect feature blocks in a cell array to avoid repeated reallocations.
    featParts = cell(1, 10);
    idx = 1;
    featParts{idx} = extractHOGFeatures(Iproc, 'CellSize', cfg.hogCellSize); idx = idx + 1;
    featParts{idx} = glcmFeatures(Iproc); idx = idx + 1;
    featParts{idx} = lbpFeaturesCompact(Iproc); idx = idx + 1;
    featParts{idx} = gaborFeatures(Iproc, cfg.gaborWavelengths, cfg.gaborOrientations); idx = idx + 1;
    featParts{idx} = edgeStats(Iproc); idx = idx + 1;
    featParts{idx} = cornerStats(Iproc); idx = idx + 1;
    featParts{idx} = intensityMoments(Iproc); idx = idx + 1;
    featParts{idx} = morphFeatures(mask); idx = idx + 1;
    featParts{idx} = colorHSVStats(I); idx = idx + 1;
    featParts{idx} = shapeFeatures(mask);
    feat = [featParts{:}];
    feat(~isfinite(feat)) = 0;
    feat = double(feat(:)');
end

function Iout = advancedPreprocessing(I, cfg)
    if size(I,3) == 3
        Ig = rgb2gray(I);
    else
        Ig = I;
    end
    Ig = im2double(Ig);
    Ig = imresize(Ig, cfg.imageSize);
    I1 = medfilt2(Ig, [3 3]);
    I2 = wiener2(I1, [5 5]);
    if cfg.useCLAHE
        I3 = adapthisteq(I2, 'ClipLimit', 0.02, 'NumTiles', [8 8], 'Distribution', 'rayleigh');
    else
        I3 = I2;
    end
    I3 = mat2gray(I3);
    if cfg.useMultiScaleFiltering
        G1 = imgaussfilt(I3, 0.5);
        G2 = imgaussfilt(I3, 1.0);
        G3 = imgaussfilt(I3, 2.0);
        I4 = (I3 + G1 + G2 + G3) / 4.0;
    else
        I4 = I3;
    end
    I5 = imsharpen(I4, 'Radius', 2, 'Amount', 0.8);
    if cfg.useMorphology
        se = strel('disk', 1);
        I6 = imclose(imopen(I5, se), se);
    else
        I6 = I5;
    end
    try
        I7 = imbilatfilt(I6, 'DegreeOfSmoothing', 0.5);
    catch
        I7 = I6;
    end
    Iout = mat2gray(I7);
end

function mask = basicSegmentationMask(Igray)
    bw = imbinarize(Igray, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.5);
    bw = bwareaopen(bw, 30);
    bw = imclose(bw, strel('disk', 2));
    mask = imfill(bw, 'holes');
end

function f = glcmFeatures(I)
    Iu = im2uint8(I);
    offsets = [0 1; -1 1; -1 0; -1 -1; 0 2; -2 2; -2 0; -2 -2];
    glcm = graycomatrix(Iu, 'Offset', offsets, 'Symmetric', true);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    c = mean(stats.Contrast);
    r = mean(stats.Correlation);
    e = mean(stats.Energy);
    h = mean(stats.Homogeneity);
    ent = zeros(1, size(glcm, 3));
    for k = 1:size(glcm, 3)
        p = glcm(:, :, k);
        p = p / (sum(p(:)) + eps);
        ent(k) = -sum(p(:) .* log2(p(:) + eps));
    end
    f = [c r e h mean(ent) std(ent)];
end

function f = lbpFeaturesCompact(I)
    try
        f = extractLBPFeatures(I, 'CellSize', [32 32], 'Normalization', 'L2');
    catch
        f = [mean(I(:)) std(I(:)) var(I(:))];
    end
end

function f = gaborFeatures(I, wavelengths, orientations)
    g = gabor(wavelengths, orientations);
    gm = imgaborfilt(I, g);
    vals = [];
    if isnumeric(gm)
        nFilters = size(gm, 3);
        vals = zeros(1, 2 * nFilters);
        for k = 1:nFilters
            M = abs(gm(:, :, k));
            idx = 2 * (k - 1) + 1;
            vals(idx:idx+1) = [mean(M(:)), std(M(:))];
        end
    elseif iscell(gm)
        nFilters = numel(gm);
        vals = zeros(1, 2 * nFilters);
        for k = 1:nFilters
            M = abs(gm{k});
            idx = 2 * (k - 1) + 1;
            vals(idx:idx+1) = [mean(M(:)), std(M(:))];
        end
    else
        vals = [0 0];
    end
    f = vals;
end

function f = edgeStats(I)
    E1 = edge(I, 'Sobel');
    E2 = edge(I, 'Canny');
    E3 = edge(I, 'log');
    [Gx, Gy] = imgradientxy(I, 'sobel');
    Gmag = hypot(Gx, Gy);
    f = [mean(E1(:)) mean(E2(:)) mean(E3(:)) mean(Gmag(:)) std(Gmag(:)) max(Gmag(:))];
end

function f = cornerStats(I)
    try
        C = detectHarrisFeatures(I);
        n = C.Count;
        if n > 0
            s = C.Metric;
            f = [n, mean(s), std(s), max(s)];
        else
            f = [0 0 0 0];
        end
    catch
        f = [0 0 0 0];
    end
end

function f = intensityMoments(I)
    v = I(:);
    f = [mean(v) var(v) skewness(v) kurtosis(v) prctile(v, 10) prctile(v, 50) prctile(v, 90)];
end

function f = morphFeatures(mask)
    try
        sk = bwmorph(mask, 'skel', Inf);
        skLen = sum(sk(:));
    catch
        skLen = 0;
    end
    eul = bweuler(mask);
    f = [skLen, eul, sum(mask(:)) / numel(mask)];
end

function f = colorHSVStats(I)
    if size(I, 3) ~= 3
        f = [0 0 0 0];
        return;
    end
    I = im2double(I);
    hsvImg = rgb2hsv(I);
    H = hsvImg(:, :, 1);
    S = hsvImg(:, :, 2);
    f = [mean(H(:)) std(H(:)) mean(S(:)) std(S(:))];
end

function f = shapeFeatures(mask)
    stats = regionprops(mask, 'Area', 'Perimeter', 'Solidity', 'Eccentricity', 'Extent');
    if isempty(stats)
        f = [0 0 0 0 0];
        return;
    end
    [~, idx] = max([stats.Area]);
    s = stats(idx);
    f = [s.Area, s.Perimeter, s.Solidity, s.Eccentricity, s.Extent];
end
