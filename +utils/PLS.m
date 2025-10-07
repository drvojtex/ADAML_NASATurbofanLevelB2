classdef PLS
    % PLS - Wrapper for MATLAB's plsregress function
    % Provides a simple fit/predict API for Partial Least Squares (PLS) regression.

    properties
        Beta        % Regression coefficients (excluding intercept)
        Intercept   % Intercept term
        MuX         % Mean of predictors
        MuY         % Mean of responses
        P           % X loadings
        Q           % Y loadings
        T           % X scores
        U           % Y scores
        W           % X weights
        ExplainX    % Explained variance in X (per component, %)
        ExplainY    % Explained variance in Y (per component, %)
    end

    methods
        function obj = fit(obj, X, Y, nComponents)
            % FIT - Train a PLS regression model using MATLAB's plsregress
            %
            % Syntax:
            %   obj = obj.fit(X, Y, nComponents)
            %
            % Inputs:
            %   X (n x p) - Predictor matrix
            %   Y (n x q) - Response matrix
            %   nComponents - Number of latent components
            %
            % Output:
            %   obj - Trained PLS model

            [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, Y, nComponents);

            obj.Beta = BETA(2:end, :);     % Regression coefficients (excluding intercept)
            obj.Intercept = BETA(1, :);    % Intercept term
            obj.MuX = mean(X, 1);
            obj.MuY = mean(Y, 1);
            obj.P = XL;
            obj.Q = YL;
            obj.T = XS;
            obj.U = YS;
            obj.W = stats.W;

            % Explained variance (as percentages)
            obj.ExplainX = 100 * PCTVAR(1, :);  % per component, in '%'
            obj.ExplainY = 100 * PCTVAR(2, :);
        end

        function Y_hat = predict(obj, X)
            % PREDICT - Predict responses using the trained PLS model
            %
            % Syntax:
            %   Y_hat = obj.predict(X)
            %
            % Inputs:
            %   X (n x p) - New predictor matrix
            %
            % Output:
            %   Y_hat (n x q) - Predicted responses

            Xc = X - obj.MuX;
            Y_hat = obj.Intercept + Xc * obj.Beta;
        end
    end
end
