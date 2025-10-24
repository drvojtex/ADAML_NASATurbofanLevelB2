classdef KernelPCR
    % KernelPCR - Principal Component Regression with a selectable Kernel (RBF or Polynomial)
    % This class implements kPCR (with intercept) for MULTIPLED OUTPUT (multi-dimensional targets Y)
    % using a SINGLE shared model based on kPCA scores.
    
    properties
        Alpha       % kPCA coefficients (n_samples x k)
        Intercept   % Intercept term (1 x n_targets)
        B_scores    % Regression coefficients for the scores T (k x n_targets) - MULTIPLED OUTPUT
        Lambda      % Eigenvalues of centered kernel matrix
        Mu          % Mean of training kernel columns (1 x n_samples)
        Mu_all      % Scalar mean of all training kernel values
        X_train     % Training predictor matrix (needed for kernel evaluation)
        
        % Kernel parameters
        KernelType  % 'rbf' or 'poly'
        Param1      % Sigma for RBF, or Degree d for Poly
        Param2      % Constant C for Poly
    end
    
    methods
        function obj = KernelPCR(kernelType, param1, param2)
            % CONSTRUCTOR: Initializes kernel type and parameters.
            
            if nargin < 1, kernelType = 'poly'; end
            if nargin < 2
                if strcmp(kernelType, 'rbf'), param1 = 1; else, param1 = 2; end
            end
            if nargin < 3, param2 = 1; end
            
            obj.KernelType = lower(kernelType);
            obj.Param1 = param1;
            obj.Param2 = param2;
        end
        
        function K = computeKernel(obj, X1, X2)
            % Compute the kernel matrix K(X1, X2)
            
            if strcmp(obj.KernelType, 'rbf')
                % RBF Kernel: K(x, z) = exp(- ||x - z||^2 / (2*sigma^2))
                sigma = obj.Param1;
                sq_dist = pdist2(X1, X2, 'squaredeuclidean');
                K = exp(-sq_dist / (2 * sigma^2));
                
            elseif strcmp(obj.KernelType, 'poly')
                % Polynomial Kernel: K(x, z) = (x*z' + C)^d
                degree = obj.Param1;
                C = obj.Param2;
                K = (X1 * X2' + C).^degree;
                
            else
                error('Unsupported kernel type. Use ''rbf'' or ''poly''.');
            end
        end
        
        function obj = fit(obj, X, Y, k)
            % FIT: Fits the kPCR model. Y is a matrix (n_samples x n_targets).
            
            [n_samples, ~] = size(X);
            
            % 1. Compute Kernel Matrix K
            K = obj.computeKernel(X, X);
            
            % 2. Center the Kernel Matrix K_c
            H = eye(n_samples) - (1/n_samples) * ones(n_samples);
            K_c = H * K * H;
            
            % 3. kPCA Eigendecomposition and sorting
            [alpha_unsorted, lambda_diag] = eig(K_c);
            [lambda_sorted, idx] = sort(diag(lambda_diag), 'descend');
            
            % Select top k components and normalize
            lambda = real(lambda_sorted(1:k));
            alpha = real(alpha_unsorted(:, idx(1:k)));
            
            for j = 1:k
                if lambda(j) > 1e-10
                    alpha(:, j) = alpha(:, j) / sqrt(lambda(j));
                else
                    alpha(:, j) = 0;
                end
            end
            
            % 4. Compute Scores T
            T_train = K_c * alpha; % n_samples x k
            
            % 5. MULTIPLED OUTPUT Regression on scores T_train
            % T_aug is [Intercept, Scores]. B_aug is the coefficients for all targets.
            T_aug = [ones(n_samples, 1), T_train];
            B_aug = T_aug \ Y;   % (k+1) x n_targets
            
            % Store coefficients
            obj.Intercept = B_aug(1,:);   % 1 x n_targets
            obj.B_scores = B_aug(2:end,:); % k x n_targets
            
            % 6. Store parameters for prediction
            obj.Alpha = alpha;
            obj.Lambda = lambda;
            obj.X_train = X;
            obj.Mu = mean(K, 1);
            obj.Mu_all = mean(obj.Mu);
        end
        
        function Y_hat = predict(obj, X_new)
            % PREDICT: Returns predicted responses Y_hat (n_new x n_targets).
            
            if isempty(obj.X_train), error('Model must be fitted before prediction.'); end
            
            n_new = size(X_new, 1);
            n_samples = size(obj.X_train, 1);
            
            % 1. Compute Test Kernel Matrix
            K_test = obj.computeKernel(X_new, obj.X_train);
            
            % 2. Center the Test Kernel Matrix
            Mu_test = mean(K_test, 2);
            
            K_test_c = K_test - Mu_test * ones(1, n_samples) ...
                              - ones(n_new, 1) * obj.Mu ...
                              + obj.Mu_all * ones(n_new, n_samples);
            
            % 3. Compute Scores T_new
            T_new = K_test_c * obj.Alpha; % n_new x k
            
            % 4. Predict responses (matrix multiplication handles all n_targets simultaneously)
            Y_hat = obj.Intercept + T_new * obj.B_scores; % n_new x n_targets
        end
    end
end