function R2 = R2eval(y, y_hat)
% R2 - Compute the coefficient of determination R^2
%
% Syntax: R2 = R2eval(y_true, y_pred)
%
% Inputs:
%   y       - vector of ground truth values
%   y_hat   - vector of predicted values
%
% Output:
%   R2      - coefficient of determination

    % Ensure inputs are column vectors
    y = y(:);
    y_hat = y_hat(:);

    % Check same length
    if length(y) ~= length(y_hat)
        error('Inputs must have the same length.');
    end

    SS_res = sum((y - y_hat).^2);
    SS_tot = sum((y - mean(y)).^2);

    R2 = 1 - (SS_res / SS_tot);
end
