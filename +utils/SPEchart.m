function SPEchart(sensorDataStd, k, alpha)
    % PCA
    [~, ~] = size(sensorDataStd);
    [coeff, score, latent] = pca(sensorDataStd);

    % Reconstruction and SPE
    Xhat = score(:,1:k) * coeff(:,1:k)';
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
    figure;
    plot(SPE, '.'); hold on;
    yline(SPEcrit, 'r--', 'LineWidth', 1.5);
    xlabel('Observations');
    ylabel('SPE (Squared Prediction Error)');
    title(sprintf('SPE (Q-statistic) Plot using first %d components', k));
    legend('SPE', 'Critical value');

    percent_below_SPE = mean(SPE < SPEcrit) * 100;
    fprintf('%.2f %% of observations are below the SPE limit.\n', percent_below_SPE);
end