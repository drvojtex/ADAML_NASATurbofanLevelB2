function T2chart(sensorDataStd, k, alpha)
    % PCA
    [n, ~] = size(sensorDataStd);
    [~, score, latent] = pca(sensorDataStd);

    % Hotelling's T2
    T2 = sum((score(:,1:k).^2) ./ latent(1:k)', 2);

    % Critical value (Hotelling T2)
    % DOI: https://doi.org/10.1016/j.cie.2019.03.021
    Fcrit = finv(alpha, k, n - k);
    T2crit = k*(n-1)*(n+1)/(n*(n-k)) * Fcrit;

    % Plot
    figure;
    plot(T2, '.'); hold on;
    yline(T2crit, 'r--', 'LineWidth', 1.5);
    xlabel('Observations');
    ylabel('Hotelling''s T^2');
    title(sprintf('Hotelling''s T^2 Plot using first %d components', k));
    legend('T^2', 'Critical value');

    percent_below_T2 = mean(T2 < T2crit) * 100;
    fprintf('%.2f %% of observations are below the Hotelling''s T^2 limit.\n', percent_below_T2);
end