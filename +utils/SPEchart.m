function SPEchart(dataTable, k, alpha)
% SPEchart - Squared Prediction Error (SPE / Q-statistic) plot for sensor data with PCA
%
% Syntax:  SPEchart(dataTable, k, alpha)
%
% Inputs:
%    dataTable - MATLAB table with sensor data. Columns containing 'Sensor'
%                are used for PCA. Must include a column 'Unit' for time series.
%    k         - Number of principal components to use in PCA.
%    alpha     - Significance level (e.g., 0.05) for SPE limit.
%
% Outputs:
%    Figure showing SPE for each Unit, critical limit line,
%    and printed percentage of observations below the limit.

    %------------------------------------------
    % Select sensor columns
    %------------------------------------------
    sensorCols = contains(dataTable.Properties.VariableNames, 'Sensor');
    sensorData = dataTable{:, sensorCols};
    units      = dataTable.Unit;
    n          = size(sensorData,1);

    %------------------------------------------
    % Robust per-unit standardization (median/MAD)
    %------------------------------------------
    sensorDataStd = zeros(size(sensorData));
    uList = unique(units);

    for u = uList'
        idx = (units == u);
        sensorDataStd(idx,:) = normalize(sensorData(idx,:), 'center', 'median', 'scale', 'mad');
    end

    %------------------------------------------
    % Remove constant or near-constant sensors
    %------------------------------------------
    varThresh = 1e-12;
    keepCols = var(sensorDataStd,0,1) > varThresh;
    if sum(~keepCols) > 0
        warning('Removing %d sensor(s) with near-zero variance.', sum(~keepCols));
    end
    sensorDataStd = sensorDataStd(:, keepCols);

    %------------------------------------------
    % Perform PCA
    %------------------------------------------
    [coeff, score, latent] = pca(sensorDataStd);

    %------------------------------------------
    % Ensure k does not exceed available PCs
    %------------------------------------------
    maxPC = min(size(score,2), rank(sensorDataStd));
    if k > maxPC
        warning('Requested k = %d is too large. Using k = %d instead.', k, maxPC);
        k = maxPC;
    end

    %------------------------------------------
    % Compute SPE (Squared Prediction Error)
    %------------------------------------------
    Xhat = score(:,1:k) * coeff(:,1:k)';
    residuals = sensorDataStd - Xhat;
    SPE = sum(residuals.^2, 2);

    %------------------------------------------
    % Saturate SPE values at 50
    %------------------------------------------
    SPE = min(SPE, 50);

    %------------------------------------------
    % Critical SPE value (Jackson & Mudholkar)
    %------------------------------------------
    theta1 = sum(latent(k+1:end));
    theta2 = sum(latent(k+1:end).^2);
    theta3 = sum(latent(k+1:end).^3);
    h0 = 1 - (2*theta1*theta3) / (3*theta2^2);
    ca = chi2inv(alpha, round((2*theta2^2)/theta3));
    SPEcrit = theta1 * ( (ca*sqrt(2*theta2*h0^2))/theta1 + 1 + ...
        (theta2*h0*(h0-1))/(theta1^2) )^(1/h0);

    %------------------------------------------
    % Plot SPE curves for each Unit
    %------------------------------------------
    figure; hold on;
    numUnits = numel(uList);
    cmap = parula(numUnits);

    for i = 1:numUnits
        idx = (units == uList(i));
        x = 1:sum(idx);
        plot(x, SPE(idx), '-', 'Color', cmap(i,:), 'LineWidth', 1.5);
    end

    % Critical SPE line
    yline(SPEcrit, 'k--', 'LineWidth', 2);

    xlabel('Observation Index within Unit');
    ylabel('SPE (Squared Prediction Error)');
    title(sprintf('SPE (Q-statistic) Plot using first %d PCs', k));

    % Colorbar for Units
    colormap(parula(numUnits));
    c = colorbar;
    c.Label.String = 'Unit Number';
    if isnumeric(uList)
        clim([uList(1), uList(end)]);
        c.Ticks = round(linspace(uList(1), uList(end), min(numUnits,5)));
    end

    %------------------------------------------
    % Percentage of observations below SPE limit
    %------------------------------------------
    percent_below_SPE = mean(SPE < SPEcrit) * 100;
    fprintf('%.2f %% of observations are below the SPE limit.\n', percent_below_SPE);

end
