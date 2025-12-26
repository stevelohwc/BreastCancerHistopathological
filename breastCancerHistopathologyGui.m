function breastCancerHistopathologyGui(modelMatPath)
% BREASTCANCERHISTOPATHOLOGYGUI
% Interactive GUI for BreastCancerHistopathological pipeline inference.
%
% Usage:
%   breastCancerHistopathologyGui
%   breastCancerHistopathologyGui('Training_results/results.mat')
%
% Notes:
% - Expects results/models saved by BreastCancerHistopathological.m:
%   `results.mat` (contains results + trained) or `models.mat` with a nearby results.mat.
% - For accurate inference the GUI needs cfg, fsInfo.keepMask, featureIndices,
%   and zParams (plus ldaModel when LDA is enabled for SVM).

    if nargin < 1 || isempty(modelMatPath)
        modelMatPath = defaultModelPath();
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

    buildUi();
    refreshModelSetUi();
    try
        addModelFile(modelMatPath, true);
    catch me
        logLine("Failed to load model file: " + string(me.message));
    end

    %% ======================== UI BUILD ========================
    function buildUi()
        app.fig = uifigure('Name', 'Breast Cancer Histopathology Classifier', ...
            'Position', [100 100 1200 720]);

        root = uigridlayout(app.fig, [2 2]);
        root.ColumnWidth = {360, '1x'};
        root.RowHeight = {'1x', 180};
        root.Padding = [10 10 10 10];

        leftPanel = uipanel(root, 'Title', 'Controls');
        leftPanel.Layout.Row = 1;
        leftPanel.Layout.Column = 1;

        left = uigridlayout(leftPanel, [4 1]);
        left.RowHeight = {'fit', 'fit', 'fit', '1x'};
        left.Padding = [8 8 8 8];

        % Model section
        modelPanel = uipanel(left, 'Title', 'Model');
        modelGrid = uigridlayout(modelPanel, [7 3]);
        modelGrid.ColumnWidth = {'fit', '1x', 'fit'};
        modelGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit'};
        modelGrid.Padding = [8 8 8 8];

        lbl = uilabel(modelGrid, 'Text', 'Model set:');
        lbl.Layout.Row = 1;
        lbl.Layout.Column = 1;
        app.ui.modelSetDropdown = uidropdown(modelGrid, ...
            'Items', {'(none)'}, 'ItemsData', 0, 'Value', 0, ...
            'ValueChangedFcn', @(~,~)onModelSetChanged());
        app.ui.modelSetDropdown.Layout.Row = 1;
        app.ui.modelSetDropdown.Layout.Column = 2;
        app.ui.btnAddModel = uibutton(modelGrid, 'Text', 'Add...', ...
            'ButtonPushedFcn', @(~,~)onAddModel());
        app.ui.btnAddModel.Layout.Row = 1;
        app.ui.btnAddModel.Layout.Column = 3;

        lbl = uilabel(modelGrid, 'Text', 'Model file:');
        lbl.Layout.Row = 2;
        lbl.Layout.Column = 1;
        app.ui.modelPath = uieditfield(modelGrid, 'text', 'Editable', 'off');
        app.ui.modelPath.Layout.Row = 2;
        app.ui.modelPath.Layout.Column = 2;
        app.ui.btnRemoveModel = uibutton(modelGrid, 'Text', 'Remove', ...
            'ButtonPushedFcn', @(~,~)onRemoveModel());
        app.ui.btnRemoveModel.Layout.Row = 2;
        app.ui.btnRemoveModel.Layout.Column = 3;

        lbl = uilabel(modelGrid, 'Text', 'Results:');
        lbl.Layout.Row = 3;
        lbl.Layout.Column = 1;
        app.ui.resultsPath = uieditfield(modelGrid, 'text', 'Editable', 'off');
        app.ui.resultsPath.Layout.Row = 3;
        app.ui.resultsPath.Layout.Column = 2;
        app.ui.btnAttachResults = uibutton(modelGrid, 'Text', 'Attach...', ...
            'ButtonPushedFcn', @(~,~)onAttachResults());
        app.ui.btnAttachResults.Layout.Row = 3;
        app.ui.btnAttachResults.Layout.Column = 3;

        lbl = uilabel(modelGrid, 'Text', 'Algorithm:');
        lbl.Layout.Row = 4;
        lbl.Layout.Column = 1;
        app.ui.modelDropdown = uidropdown(modelGrid, ...
            'Items', {'SVM', 'Random Forest', 'XGBoost'}, ...
            'Value', 'SVM', ...
            'ValueChangedFcn', @(~,~)onModelChanged());
        app.ui.modelDropdown.Layout.Row = 4;
        app.ui.modelDropdown.Layout.Column = [2 3];

        lbl = uilabel(modelGrid, 'Text', 'Options:');
        lbl.Layout.Row = 5;
        lbl.Layout.Column = 1;
        app.ui.autoPredict = uicheckbox(modelGrid, 'Text', 'Auto', 'Value', true);
        app.ui.autoPredict.Layout.Row = 5;
        app.ui.autoPredict.Layout.Column = 2;
        app.ui.compareAll = uicheckbox(modelGrid, 'Text', 'Compare', 'Value', false, ...
            'ValueChangedFcn', @(~,~)onCompareChanged());
        app.ui.compareAll.Layout.Row = 5;
        app.ui.compareAll.Layout.Column = 3;

        app.ui.lblBundle = uilabel(modelGrid, 'Text', 'Bundle: -', ...
            'FontColor', [0.5 0.5 0.5]);
        app.ui.lblBundle.Layout.Row = 6;
        app.ui.lblBundle.Layout.Column = [1 3];
        app.ui.lblBundle.Tooltip = 'Bundle readiness for inference.';

        app.ui.lblOverall = uilabel(modelGrid, 'Text', 'Overall: -', ...
            'FontColor', [0.25 0.25 0.25], 'FontSize', 11);
        app.ui.lblOverall.Layout.Row = 7;
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

        % Predict section
        predPanel = uipanel(left, 'Title', 'Prediction');
        predGrid = uigridlayout(predPanel, [2 1]);
        predGrid.ColumnWidth = {'1x'};
        predGrid.RowHeight = {'fit', '1x'};
        predGrid.Padding = [8 8 8 8];

        app.ui.btnPredict = uibutton(predGrid, 'Text', 'Predict', ...
            'ButtonPushedFcn', @(~,~)onPredict());
        app.ui.btnPredict.Layout.Row = 1;
        app.ui.btnPredict.Layout.Column = 1;

        predMetrics = uigridlayout(predGrid, [3 4]);
        predMetrics.Layout.Row = 2;
        predMetrics.Layout.Column = 1;
        predMetrics.ColumnWidth = {'fit', '1x', 'fit', '1x'};
        predMetrics.RowHeight = {'fit', 'fit', 'fit'};
        predMetrics.Padding = [0 0 0 0];

        lbl = uilabel(predMetrics, 'Text', 'Using:');
        lbl.Layout.Row = 1;
        lbl.Layout.Column = 1;
        app.ui.predAlg = uilabel(predMetrics, 'Text', '-', 'HorizontalAlignment', 'right');
        app.ui.predAlg.Layout.Row = 1;
        app.ui.predAlg.Layout.Column = 2;

        lbl = uilabel(predMetrics, 'Text', 'Benign:');
        lbl.Layout.Row = 1;
        lbl.Layout.Column = 3;
        app.ui.scoreBenign = uilabel(predMetrics, 'Text', '-', 'HorizontalAlignment', 'right');
        app.ui.scoreBenign.Layout.Row = 1;
        app.ui.scoreBenign.Layout.Column = 4;

        lbl = uilabel(predMetrics, 'Text', 'Label:');
        lbl.Layout.Row = 2;
        lbl.Layout.Column = 1;
        app.ui.predLabel = uilabel(predMetrics, 'Text', '-', 'HorizontalAlignment', 'right');
        app.ui.predLabel.Layout.Row = 2;
        app.ui.predLabel.Layout.Column = 2;

        lbl = uilabel(predMetrics, 'Text', 'Malignant:');
        lbl.Layout.Row = 2;
        lbl.Layout.Column = 3;
        app.ui.scoreMalignant = uilabel(predMetrics, 'Text', '-', 'HorizontalAlignment', 'right');
        app.ui.scoreMalignant.Layout.Row = 2;
        app.ui.scoreMalignant.Layout.Column = 4;

        lbl = uilabel(predMetrics, 'Text', 'Confidence:');
        lbl.Layout.Row = 3;
        lbl.Layout.Column = 1;
        app.ui.predConf = uilabel(predMetrics, 'Text', '-', 'HorizontalAlignment', 'right');
        app.ui.predConf.Layout.Row = 3;
        app.ui.predConf.Layout.Column = 2;

        % Right side (preview + scores)
        rightPanel = uipanel(root, 'Title', 'Preview');
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
        app.ui.axScores = uiaxes(scoresGrid);
        app.ui.axScores.Layout.Row = 1;
        title(app.ui.axScores, 'Per-Class Scores');
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

        % Log (span full width, fixed height)
        logPanel = uipanel(root, 'Title', 'Log');
        logPanel.Layout.Row = 2;
        logPanel.Layout.Column = [1 2];
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
        [file, path] = uigetfile('*.mat', 'Select model/results .mat file');
        if isequal(file, 0)
            return;
        end
        addModelFile(fullfile(path, file), true);
    end

    function onRemoveModel()
        if isempty(app.modelSets) || app.activeModelIdx < 1
            return;
        end
        removedName = string(app.modelSets(app.activeModelIdx).displayName);
        app.modelSets(app.activeModelIdx) = [];

        if isempty(app.modelSets)
            app.activeModelIdx = 0;
            app.model = struct();
            refreshModelSetUi();
        else
            app.activeModelIdx = min(app.activeModelIdx, numel(app.modelSets));
            setActiveModelIdx(app.activeModelIdx);
        end

        logLine("Removed model set: " + removedName);
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

    function onAttachResults()
        if isempty(app.modelSets) || app.activeModelIdx < 1
            return;
        end
        [file, path] = uigetfile('results*.mat', 'Select results.mat');
        if isequal(file, 0)
            return;
        end
        try
            attachResultsFile(fullfile(path, file));
        catch me
            logLine("Attach results failed: " + string(me.message));
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

        if app.ui.autoPredict.Value
            onPredict();
        end
    end

    function onClearImage()
        app.currentImage = [];
        app.currentImagePath = '';
        app.ui.imagePath.Value = '';
        cla(app.ui.axImage);
        cla(app.ui.axScores);
        if isfield(app.ui, 'tblCompare') && ~isempty(app.ui.tblCompare)
            app.ui.tblCompare.Data = cell(0, 5);
        end
        app.ui.predAlg.Text = '-';
        app.ui.predLabel.Text = '-';
        app.ui.predConf.Text = '-';
        app.ui.scoreBenign.Text = '-';
        app.ui.scoreMalignant.Text = '-';
        logLine('Cleared image.');
    end

    function onModelChanged()
        if isfield(app.ui, 'compareAll') && app.ui.compareAll.Value
            if app.ui.autoPredict.Value && ~isempty(app.currentImage)
                onPredict();
            end
            return;
        end
        if app.ui.autoPredict.Value && ~isempty(app.currentImage)
            onPredict();
        end
    end

    function onCompareChanged()
        if app.uiUpdating
            return;
        end
        if app.ui.compareAll.Value
            app.ui.modelDropdown.Enable = 'off';
        else
            app.ui.modelDropdown.Enable = 'on';
        end
        if app.ui.autoPredict.Value && ~isempty(app.currentImage)
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

            isCompare = isfield(app.ui, 'compareAll') && app.ui.compareAll.Value;
            if isCompare
                results = predictAllAlgorithms(app.currentImage);
            else
                results = predictAlgorithmSafe(app.currentImage, app.ui.modelDropdown.Value);
            end

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
        app.ui.modelPath.Value = app.model.path;
        app.ui.resultsPath.Value = app.model.resultsPath;

        updateBundleStatus();
        updateOverallStatus();
        app.uiUpdating = false;
        if isfield(app.ui, 'compareAll') && app.ui.compareAll.Value
            app.ui.modelDropdown.Enable = 'off';
        else
            app.ui.modelDropdown.Enable = 'on';
        end

        if app.ui.autoPredict.Value && ~isempty(app.currentImage) && app.bundleReady
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
            app.ui.modelPath.Value = '';
            app.ui.resultsPath.Value = '';
            app.ui.modelDropdown.Enable = 'off';
            app.ui.autoPredict.Enable = 'off';
            app.ui.compareAll.Enable = 'off';
            app.ui.btnRemoveModel.Enable = 'off';
            app.ui.btnPredict.Enable = 'off';
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
        app.ui.btnRemoveModel.Enable = 'on';
        app.ui.btnPredict.Enable = 'on';
        app.ui.autoPredict.Enable = 'on';
        app.ui.compareAll.Enable = 'on';
        app.uiUpdating = wasUpdating;
    end

    function updateBundleStatus()
        [ready, statusText, color] = evaluateBundleStatus(app.model);
        app.bundleReady = ready;
        app.ui.lblBundle.Text = statusText;
        app.ui.lblBundle.FontColor = color;

        if ready
            app.ui.btnPredict.Enable = 'on';
            app.ui.autoPredict.Enable = 'on';
            app.ui.compareAll.Enable = 'on';
        else
            app.ui.btnPredict.Enable = 'off';
            app.ui.autoPredict.Enable = 'off';
            app.ui.compareAll.Enable = 'off';
            app.ui.modelDropdown.Enable = 'off';
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

                app.ui.lblOverall.Text = "Overall best (test Sens): " + algName + " (" + sprintf('%.1f%%', bestSens * 100) + ")";
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

    function results = predictAllAlgorithms(I)
        algs = {'SVM', 'Random Forest', 'XGBoost'};
        first = predictAlgorithmSafe(I, algs{1});
        results = repmat(first, 1, numel(algs));
        for i = 2:numel(algs)
            results(i) = predictAlgorithmSafe(I, algs{i});
        end
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
            best = results(bestIdx);
            app.ui.predAlg.Text = best.algorithm;
            app.ui.predLabel.Text = best.label;
            app.ui.predConf.Text = fmtScore(best.conf);
            app.ui.scoreBenign.Text = fmtScore(best.benign);
            app.ui.scoreMalignant.Text = fmtScore(best.malignant);
            showScores(best.classNames, best.scores);
        else
            app.ui.predAlg.Text = '-';
            app.ui.predLabel.Text = '-';
            app.ui.predConf.Text = '-';
            app.ui.scoreBenign.Text = '-';
            app.ui.scoreMalignant.Text = '-';
            cla(app.ui.axScores);
        end
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

    function showScores(classNames, scores)
        cla(app.ui.axScores);
        if isempty(scores) || isempty(classNames)
            title(app.ui.axScores, 'Per-Class Scores');
            return;
        end
        scores = double(scores(:)');
        classNames = string(classNames(:)');
        if numel(scores) ~= numel(classNames)
            n = min(numel(scores), numel(classNames));
            scores = scores(1:n);
            classNames = classNames(1:n);
        end
        bar(app.ui.axScores, scores);
        app.ui.axScores.XTick = 1:numel(scores);
        app.ui.axScores.XTickLabel = cellstr(string(classNames));
        app.ui.axScores.YLim = [0 1];
        ylabel(app.ui.axScores, 'Normalized score');
        grid(app.ui.axScores, 'on');
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
    p = 'results.mat';
    if ~isfile(p)
        candidate = fullfile(pwd, 'Training_results', 'results.mat');
        if isfile(candidate)
            p = candidate;
        end
    end
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
        for k = 1:size(gm, 3)
            M = abs(gm(:, :, k));
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
