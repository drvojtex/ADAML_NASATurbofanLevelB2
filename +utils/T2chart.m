function T2chart(dataTable, k, alpha)
% T2chart - Hotelling's T² plot for sensor data with PCA
%
% Syntax:  T2chart(dataTable, k, alpha)
%
% Inputs:
%    dataTable - MATLAB table with sensor data. Columns containing 'Sensor'
%                are used for PCA. Must include a column 'Unit' for time series.
%    k         - Number of principal components to use in PCA.
%    alpha     - Significance level (e.g., 0.05) for Hotelling's T² limit.
%
% Outputs:
%    Figure showing Hotelling's T² for each Unit, critical limit line,
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
        % Robust median/MAD scaling per feature within the Unit
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
    [~, score, latent] = pca(sensorDataStd);

    %------------------------------------------
    % Ensure k does not exceed available PCs
    %------------------------------------------
    maxPC = min(size(score,2), rank(sensorDataStd));
    if k > maxPC
        warning('Requested k = %d is too large. Using k = %d instead.', k, maxPC);
        k = maxPC;
    end

    %------------------------------------------
    % Compute Hotelling's T²
    %------------------------------------------
    T2 = sum((score(:,1:k).^2) ./ latent(1:k)', 2);

    %------------------------------------------
    % Saturate T² values at 50
    %------------------------------------------
    T2 = min(T2, 50);

    % Critical Hotelling's T² value
    Fcrit = finv(1 - alpha, k, n - k);
    T2crit = k * (n - 1) / (n - k) * Fcrit;

    %------------------------------------------
    % Plot T² curves for each Unit
    %------------------------------------------
    figure; hold on;
    numUnits = numel(uList);
    cmap = parula(numUnits);  % color gradient

    for i = 1:numUnits
        idx = (units == uList(i));
        x = 1:sum(idx);  % index within Unit
        plot(x, T2(idx), '-', 'Color', cmap(i,:), 'LineWidth', 1.5);
    end

    % Critical T² line
    yline(T2crit, 'k--', 'LineWidth', 2);

    xlabel('Observation Index within Unit');
    ylabel('Hotelling''s T^2');
    title(sprintf('Hotelling''s T^2 Plot (k = %d PCs)', k));

    % Colorbar for Units
    colormap(parula(numUnits));
    c = colorbar;
    c.Label.String = 'Unit Number';
    if isnumeric(uList)
        clim([uList(1), uList(end)]);
        c.Ticks = round(linspace(uList(1), uList(end), min(numUnits,5)));
    end

    %------------------------------------------
    % Percentage of observations below T² limit
    %------------------------------------------
    percent_below_T2 = mean(T2 < T2crit) * 100;
    fprintf('%.2f %% of observations are below the Hotelling''s T^2 limit.\n', ...
            percent_below_T2);
end
