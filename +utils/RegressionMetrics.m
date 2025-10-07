classdef RegressionMetrics
    %REGRESSIONMETRICS Class for evaluating regression model performance
    %
    %   metrics = RegressionMetrics(y_true, y_pred, y_train)
    %
    %   Inputs:
    %       y_true  - matrix of true values [n_samples x n_targets]
    %       y_pred  - matrix of predicted values [n_samples x n_targets]
    %       y_train - (optional) training data values [n_train x n_targets]
    %                 used for computing Q2
    %
    %   Methods:
    %       computeRMSE()  - returns RMSE per output dimension
    %       computeR2()    - returns R2 per output dimension
    %       computeQ2()    - returns Q2 per output dimension
    %       summary()      - prints all metrics in a readable format

    properties (SetAccess = private)
        y_true double
        y_pred double
        y_train double = []
    end

    methods
        %% --- Constructor ---
        function obj = RegressionMetrics(y_true, y_pred, y_train)
            arguments
                y_true (:,:) double
                y_pred (:,:) double
                y_train (:,:) double = []
            end

            if ~isequal(size(y_true), size(y_pred))
                error('The sizes of y_true and y_pred must match.');
            end

            if ~isempty(y_train) && size(y_true, 2) ~= size(y_train, 2)
                error('The number of target dimensions in y_true and y_train must match.');
            end

            obj.y_true = y_true;
            obj.y_pred = y_pred;
            obj.y_train = y_train;
        end

        %% --- RMSE ---
        function rmse = computeRMSE(obj)
            % Computes Root Mean Squared Error (RMSE) per target dimension
            errors = obj.y_true - obj.y_pred;
            rmse = sqrt(mean(errors.^2, 1));
        end

        %% --- R2 ---
        function R2 = computeR2(obj)
            % Computes the coefficient of determination (R2)
            SS_res = sum((obj.y_true - obj.y_pred).^2, 1);
            SS_tot = sum((obj.y_true - mean(obj.y_true, 1)).^2, 1);
            R2 = 1 - (SS_res ./ SS_tot);
        end

        %% --- Q2 ---
        function Q2 = computeQ2(obj)
            % Computes the predictive squared correlation coefficient (Q2)
            if isempty(obj.y_train)
                error('y_train must be provided to compute Q².');
            end

            y_train_mean = mean(obj.y_train, 1);
            SS_res = sum((obj.y_true - obj.y_pred).^2, 1);
            SS_pred = sum((obj.y_true - y_train_mean).^2, 1);
            Q2 = 1 - (SS_res ./ SS_pred);
        end

        %% --- Summary ---
        function summary(obj, comment)
            fprintf('--- %s Regression Metrics ---\n', comment);
            rmse = mean(obj.computeRMSE());
            fprintf('RMSE:\t%s\n', mat2str(rmse, 4));
            
            if isempty(obj.y_train)
                R2   = mean(obj.computeR2());
                fprintf('R2:\t%s\n', mat2str(R2, 4));
            end

            if ~isempty(obj.y_train)
                Q2 = mean(obj.computeQ2());
                fprintf('Q2:\t%s\n', mat2str(Q2, 4));
            end
        end
    end
end
