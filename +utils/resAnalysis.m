function resAnalysis(y, y_hat)
% resAnalysis - Plot residual vs. time, residual histogram, residual
% autolag (50 lags) and a observed vs. predicted plot.
%
% Syntax:  resAnalysis(y, y_hat)
%
% Inputs:
%    y      - Observed/true values of y as a vector
%    y_hat  - Predicted values of y as a vector.
%
% Outputs:
%   Plots for residual analysis (residual vs. time, residual histogram, 
%   residual autolag (50 lags) and a observed vs. predicted plot)

res = y -y_hat;

% Calculate confidence interval
confLevel = 0.95;
alpha = (1 - confLevel) / 2;
lower_q = alpha;
upper_q = 1 - alpha;
lowerBound = quantile(res, lower_q, 2);
upperBound = quantile(res, upper_q, 2);
medianRes  = median(res, 2);

figure('Name','Residual analysis (test set)','Position',[100 100 1050 350]);

subplot(1,3,1)
plot(1:length(res), res,'-','LineWidth',1.2)
yline(lowerBound, '--k', 'Lower', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
yline(medianRes, '--k', 'Median', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
yline(upperBound, '--k', 'Upper', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
xlabel('Time'); ylabel('Residual'); grid on
title('Residuals vs. time')

subplot(1,3,2)
histogram(res, 20,'FaceColor',[0.2 0.6 0.8])
xline(lowerBound, '--k', 'Lower', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
xline(medianRes, '--k', 'Median', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
xline(upperBound, '--k', 'Upper', 'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
xlabel('Residual'); ylabel('Frequency'); grid on
title('Residual histogram')

subplot(1,3,3)
% Simple ACF of residuals
maxLag = 50;
res_c = res - mean(res);
acf = xcorr(res_c, maxLag, 'coeff');
stem(0:maxLag, acf(maxLag+1:end), 'filled')
xlabel('Lag'); ylabel('ACF'); grid on
title('Residual autocorrelation')
yline(0,'k-'); 

figure;
scatter(y, y_hat, 15, 'filled'); hold on
plot(xlim, xlim,'k--','LineWidth',1.2); hold off
xlabel('Observed y (test)'); ylabel('Predicted y (test)');
title('Observed vs Predicted'); grid on; axis equal tight