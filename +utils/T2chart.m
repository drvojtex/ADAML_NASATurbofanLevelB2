function T2chart(dataTable, k, alpha)
% T2chart - Hotelling's T² plot for sensor data with PCA
%
% Syntax:  T2chart(dataTable, k, alpha)
%
% Inputs:
%    dataTable - MATLAB table containing the dataset. Columns with 'Sensor' in
%                the name are used for PCA. Must include a column 'Unit'.
%    k         - Number of principal components to use in PCA
%    alpha     - Significance level for Hotelling's T² critical value
%
% Outputs:
%    This function produces a figure showing Hotelling's T² for each Unit
%    as colored curves, with a critical T² line. It also prints the percentage
%    of observations below the critical limit.

    % Select columns containing 'Sensor'
    sensorCols = contains(dataTable.Properties.VariableNames, 'Sensor');
    sensorData = dataTable{:, sensorCols};

    % Standardize sensor data (z-score)
    sensorDataStd = zscore(sensorData);

    % Perform PCA
    [n, ~] = size(sensorDataStd);
    [~, score, latent] = pca(sensorDataStd);

    % Compute Hotelling's T²
    T2 = sum((score(:,1:k).^2) ./ latent(1:k)', 2);

    % Compute critical value for Hotelling's T²
    % DOI: https://doi.org/10.1016/j.cie.2019.03.021
    Fcrit = finv(alpha, k, n - k);
    T2crit = k*(n-1)*(n+1)/(n*(n-k)) * Fcrit;

    % Create figure and hold for multiple curves
    figure; hold on;
    units = unique(dataTable.Unit);
    numUnits = numel(units);

    % Create a color gradient like a heatmap
    cmap = parula(numUnits);  % parula provides a smooth gradient

    % Plot each Unit as a separate curve with color from gradient
    for i = 1:numUnits
        idx = dataTable.Unit == units(i);
        x = 1:sum(idx);  % X-axis = index within the Unit
        plot(x, T2(idx), '-', 'Color', cmap(i,:), 'LineWidth', 1.5);
    end

    % Plot critical T² value as dashed black line
    yline(T2crit, 'k--', 'LineWidth', 2);

    % Label axes and title
    xlabel('Observation Index within Unit');
    ylabel('Hotelling''s T^2');
    title(sprintf('Hotelling''s T^2 Plot using first %d components', k));

    % Add colorbar to indicate Unit number
    colormap(parula(numUnits));
    c = colorbar;
    c.Label.String = 'Unit Number';
    clim([units(1), units(end)]); % Map first to last Unit to color gradient
    c.Ticks = round(linspace(units(1), units(end), 5)); % e.g., 5 tick labels

    % Compute and display percentage of observations below the T² limit
    percent_below_T2 = mean(T2 < T2crit) * 100;
    fprintf('%.2f %% of observations are below the Hotelling''s T^2 limit.\n', percent_below_T2);
end
