classdef PCR
    % PCR - Principal Component Regression class
    % This class implements PCR (with intercept) for multi-dimensional targets.
    % X is automatically centered via PCA. Intercept is estimated in regression.
    
    properties
        Beta        % Regression coefficients in original feature space (n_features x n_targets)
        Intercept   % Intercept term (1 x n_targets)
        Coeff       % PCA loadings (n_features x k)
        Latent      % eigenvalues of covariance matrix
        Explained   % percentage ratio of explained variance for each component
        Mu          % Mean of X (1 x n_features) for centering new data
    end
    
    methods
        function obj = fit(obj, X, Y, k)
            % FIT Fit the PCR model to the data
            % Inputs:
            %   X - Predictor matrix (n_samples x n_features)
            %   Y - Response matrix (n_samples x n_targets)
            %   k - Number of latent variables (principal components)
            
            % Perform PCA (MATLAB automatically centers X)
            [coeff, score, latent, ~, explained, mu] = pca(X, 'NumComponents', k);
            
            % Add intercept term to regression on scores
            T_aug = [ones(size(score,1),1), score];
            B_aug = T_aug \ Y;   % (k+1) x n_targets
            
            % Separate intercept and score coefficients
            intercept = B_aug(1,:);
            B_scores = B_aug(2:end,:);
            
            % Map back to original feature space
            beta = coeff * B_scores;
            
            % Store in object
            obj.Beta = beta;
            obj.Intercept = intercept;
            obj.Coeff = coeff;
            obj.Latent = latent;
            obj.Explained = explained;
            obj.Mu = mu;
        end
        
        function Y_hat = predict(obj, X)
            % PREDICT Predict responses for new data
            % Inputs:
            %   X - New predictor matrix (n_samples x n_features)
            % Outputs:
            %   Y_hat - Predicted response matrix (n_samples x n_targets)
            
            % Center new data using training mean
            X_centered = X - obj.Mu;
            
            % Compute predictions
            Y_hat = obj.Intercept + X_centered * obj.Beta;
        end
    end
end
