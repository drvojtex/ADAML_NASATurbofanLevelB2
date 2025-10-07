function plot_curve_reconstruction(Y_true, Y_pred, title_str)
    % PLOT_CURVE_RECONSTRUCTION
    %   Reconstructs full curves from consecutive segments (N, f)
    %   and plots both true and predicted data for visual comparison.
    %
    %   Parameters:
    %       Y_true   - [N, f] matrix of true data
    %       Y_pred   - [N, f] matrix of predicted data
    %       title_str - title of the plot

    if nargin < 3
        error('Usage: plot_curve_reconstruction(Y_true, Y_pred, title_str)');
    end
    if ~isequal(size(Y_true), size(Y_pred))
        error('Y_true and Y_pred must have the same dimensions.');
    end

    % Concatenate segments to reconstruct the curve
    curve_true = reshape(Y_true', 1, []);  % transpose then linearize
    curve_pred = reshape(Y_pred', 1, []);

    % Time axis
    len = length(curve_true);
    t = 1:len;

    % Plot
    figure;
    plot(t, curve_true, 'r', 'LineWidth', 1.5); hold on;
    plot(t, curve_pred, 'b', 'LineWidth', 1.5);
    xlabel('Time');
    ylabel('Sensor value');
    title(title_str, 'Interpreter', 'none');
    legend('True', 'Prediction', 'Location', 'northwest');
    grid on;
end
