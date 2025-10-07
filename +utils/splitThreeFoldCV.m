function [train_X, train_Y, calib_X, calib_Y] = splitThreeFoldCV(X, Y, fold)
%SPLITTHREEFOLDCV 3-fold cross-validation split
%
%   [train_X, train_Y, calib_X, calib_Y] = splitThreeFoldCV(X, Y, fold)
%
%   Inputs:
%       X     - input matrix [N × f]
%       Y     - target matrix [N × t] or vector [N × 1]
%       fold  - integer {1, 2, 3}, specifies which fold is used for validation
%
%   Outputs:
%       train_X, train_Y - training data (2/3 of samples)
%       calib_X, calib_Y - calibration (validation) data (1/3 of samples)
%
%   The data are split sequentially into 3 equal parts (±1 sample if N not divisible by 3).

    arguments
        X (:,:) double
        Y (:,:) double
        fold (1,1) {mustBeMember(fold, [1, 2, 3])}
    end

    N = size(X, 1);
    n1 = floor(N / 3);
    n2 = floor(2 * N / 3);

    % Define folds
    X_fold_1 = X(1:n1, :);
    Y_fold_1 = Y(1:n1, :);

    X_fold_2 = X(n1+1:n2, :);
    Y_fold_2 = Y(n1+1:n2, :);

    X_fold_3 = X(n2+1:end, :);
    Y_fold_3 = Y(n2+1:end, :);

    switch fold
        case 1
            calib_X = X_fold_1; calib_Y = Y_fold_1;
            train_X = [X_fold_2; X_fold_3];
            train_Y = [Y_fold_2; Y_fold_3];
        case 2
            calib_X = X_fold_2; calib_Y = Y_fold_2;
            train_X = [X_fold_1; X_fold_3];
            train_Y = [Y_fold_1; Y_fold_3];
        case 3
            calib_X = X_fold_3; calib_Y = Y_fold_3;
            train_X = [X_fold_1; X_fold_2];
            train_Y = [Y_fold_1; Y_fold_2];
    end
end
