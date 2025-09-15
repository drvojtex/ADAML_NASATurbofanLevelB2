function SPEchart(dataTable, k, alpha)
% SPEchart - Squared Prediction Error (SPE / Q-statistic) plot for sensor data with PCA
%
% Syntax:  SPEchart(dataTable, k, alpha)
%
% Inputs:
%    dataTable - MATLAB table containing the dataset. Columns with 'Sensor' in
%                the name are used for PCA. Must include a column 'Unit'.
%    k         - Number of principal components to use in PCA
%    alpha     - Significance level for SPE critical value
%
% Outputs:
%    This function produces a figure showing SPE for each Unit
%    as colored curves, with a critical SPE line. It also prints the percentage
%    of observations below the critical limit.

    % Select columns containing 'Sensor'
    sensorCols = contains(dataTable.Properties.VariableNames, 'Sensor');
    sensorData = dataTable{:, sensorCols};

    % Standardize sensor data (z-score)
    sensorDataStd = zscore(sensorData);

    % Perform PCA
    [coeff, score, latent] = pca(sensorDataStd);

    % Compute SPE (Squared Prediction Error)
    Xhat = score(:,1:k) * coeff(:,1:k)';  % Reconstruction using first k PCs
    residuals = sensorDataStd - Xhat;
    SPE = sum(residuals.^2, 2);

    % Critical value (Jackson & Mudholkar)
    % DOI: https://doi.org/10.1080/00401706.1979.10489779
    % DOI: https://doi.org/10.1016/j.eswa.2012.12.020
    theta1 = sum(latent(k+1:end));
    theta2 = sum(latent(k+1:end).^2);
    theta3 = sum(latent(k+1:end).^3);
    h0 = 1 - (2*theta1*theta3) / (3*theta2^2);
    ca = chi2inv(alpha, round((2*theta2^2)/theta3));
    SPEcrit = theta1 * ( (ca*sqrt(2*theta2*h0^2))/theta1 + 1 + ...
        (theta2*h0*(h0-1))/(theta1^2) )^(1/h0);

    % Plot
    figure; hold on;
    units = unique(dataTable.Unit);
    numUnits = numel(units);

    % Color gradient for multiple Units
    cmap = parula(numUnits);

    % Plot each Unit as a separate curve
    for i = 1:numUnits
        idx = dataTable.Unit == units(i);
        x = 1:sum(idx);  % X-axis = observation index within Unit
        plot(x, SPE(idx), '-', 'Color', cmap(i,:), 'LineWidth', 1.5);
    end

    % Plot critical SPE value as dashed black line
    yline(SPEcrit, 'k--', 'LineWidth', 2);

    % Label axes and title
    xlabel('Observation Index within Unit');
    ylabel('SPE (Squared Prediction Error)');
    title(sprintf('SPE (Q-statistic) Plot using first %d components', k));

    % Colorbar to indicate Unit number
    colormap(parula(numUnits));
    c = colorbar;
    c.Label.String = 'Unit Number';
    clim([units(1), units(end)]);
    c.Ticks = round(linspace(units(1), units(end), 5));

    % Compute and display percentage of observations below the SPE limit
    percent_below_SPE = mean(SPE < SPEcrit) * 100;
    fprintf('%.2f %% of observations are below the SPE limit.\n', percent_below_SPE);
end




