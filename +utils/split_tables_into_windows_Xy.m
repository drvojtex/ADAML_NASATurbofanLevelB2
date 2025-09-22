function [X, y, standard_params] = split_tables_into_windows_Xy(tablesArray, window_size, stride, target_column, mu_sigma)
% SPLIT_TABLES_INTO_WINDOWS_XY
% Split tables into sliding windows for features and target,
% append linear fit coefficients, and optionally z-score standardize
% columns containing 'Sensor' in their names.
%
%   [X, y, standard_params] = SPLIT_TABLES_INTO_WINDOWS_XY(...,
%               tablesArray, window_size, stride, target_column, mu_sigma)
%
%   Inputs:
%       tablesArray   - cell array of tables to be split
%       window_size   - number of rows in each sliding window
%       stride        - step size to move the window along the table
%       target_column - name of the column to use as target variable
%       mu_sigma      - optional struct with fields:
%                         mu.<SensorColumnName>
%                         sigma.<SensorColumnName>
%                       If provided, these are used for z-score standardization.
%
%   Outputs:
%       X - 3D matrix of size (num_windows, window_size+2, num_features-1)
%       y - 3D matrix of size (num_windows, window_size, 1)
%       standard_params - struct with fields mu and sigma for each Sensor column

    % --- Estimate total number of windows for preallocation ---
    totalWindows = 0;
    for i = 1:length(tablesArray)
        nRows = height(tablesArray{i});
        totalWindows = totalWindows + max(floor((nRows - window_size)/stride) + 1, 0);
    end

    % --- Determine number of features (excluding target) ---
    num_features = width(tablesArray{1});
    feature_names = tablesArray{1}.Properties.VariableNames;

    % --- Identify Sensor columns ---
    sensor_cols = contains(feature_names, 'Sensor');
    sensor_col_names = feature_names(sensor_cols);
    
    % Initialize X and y
    X = zeros(totalWindows, window_size + 2, num_features - 1);
    y = zeros(totalWindows, window_size, 1);
    
    % Initialize struct for storing mu and sigma
    standard_params = struct('mu', struct(), 'sigma', struct());

    % --- Compute mu/sigma if not provided ---
    if nargin < 5 || isempty(mu_sigma)
        all_sensor_data = [];
        for i = 1:length(tablesArray)
            T = tablesArray{i};
            all_sensor_data = [all_sensor_data; table2array(T(:, sensor_cols))];
        end
        mu_vals = mean(all_sensor_data, 1);
        sigma_vals = std(all_sensor_data, 0, 1);
        for s = 1:length(sensor_col_names)
            standard_params.mu.(sensor_col_names{s}) = mu_vals(s);
            standard_params.sigma.(sensor_col_names{s}) = sigma_vals(s);
        end
    else
        standard_params = mu_sigma;
    end

    k = 1; % global window index across all tables

    % --- Loop through all tables ---
    for i = 1:length(tablesArray)
        T = tablesArray{i};
        nRows = height(T);

        % Precompute indexes of target/features
        target_idx = find(strcmp(T.Properties.VariableNames, target_column));
        feature_idx = setdiff(1:num_features, target_idx);

        % Keep previous window features for linear fit
        prev_window_features = [];

        % Loop through table with the given stride
        for startRow = 1:stride:(nRows - window_size + 1)
            endRow = startRow + window_size - 1;

            % Extract current window
            window = T(startRow:endRow, :);
            features = table2array(window(:, feature_idx));
            target   = table2array(window(:, target_idx));

            % --- Z-score standardization for Sensor columns ---
            sensor_idx_in_window = find(sensor_cols(feature_idx));
            for s = 1:length(sensor_idx_in_window)
                idx = sensor_idx_in_window(s);
                col_name = feature_names{feature_idx(idx)};
                mu = standard_params.mu.(col_name);
                sigma = standard_params.sigma.(col_name);
                features(:, idx) = (features(:, idx) - mu) / sigma;
            end

            % --- Prepare augmented feature window ---
            augmented = zeros(window_size + 2, num_features - 1);
            augmented(1:window_size, :) = features;

            if isempty(prev_window_features)
                % First window in this table: zeros
                % (no previous window available)
                % already initialized to zero
            else
                % Compute slope (a) and intercept (b) for each feature
                x = (1:window_size)';
                a = zeros(1, num_features - 1);
                b = zeros(1, num_features - 1);
                for f = 1:(num_features - 1)
                    p = polyfit(x, prev_window_features(:, f), 1);
                    a(f) = p(1);
                    b(f) = p(2);
                end
                augmented(window_size + 1, :) = a;
                augmented(window_size + 2, :) = b;
            end

            % Store into X and y
            X(k, :, :) = augmented;
            y(k, :, 1) = target;

            % Update previous window (for next iteration)
            prev_window_features = features;

            k = k + 1;
        end
    end
end
