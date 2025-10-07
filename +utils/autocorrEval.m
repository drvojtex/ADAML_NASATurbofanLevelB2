function [acf, lags] = autocorrEval(y, y_hat, maxLag)
% autocorrEval - Plots figure showing autocorrelation of prediction
% residuals and  returns the sample autocorrelation function and
% associated lags
%
% Syntax:  [acf, lags] = autocorrEval(y, y_hat, maxLag)
%
% Inputs:
%   y       - vector of ground truth values
%   y_hat   - vector of predicted values
%   maxLag  - (optional) maximum lag to use for autocorrelation
%             (default = 20)
%
% Outputs:
%   Figure showing autocorrelation of prediction residuals.
%   acf     - sample autocorrelation function
%   lags    - associated lags

    if nargin < 3
        maxLag = 20;
    end

    % Ensure inputs are column vectors
    y = y(:);
    y_hat = y_hat(:);

    if length(y) ~= length(y_hat)
        error('Inputs must have the same length.');
    end

    % Compute residuals
    res = y - y_hat;

    % --- Plot autocorrelation function (ACF) ---
    figure;
    % Simple ACF of residuals
    res_c = res - mean(res);
    acf = xcorr(res_c, maxLag, 'coeff');
    lags = 0:maxLag;
    stem(lags, acf(maxLag+1:end), 'filled')
    xlabel('Lag'); ylabel('ACF'); grid on
    title('Residual autocorrelation')
end
