function df = dataloader(filename)
    %NASA-Turbofan-data: Load engine dataset into a MATLAB table.
    %
    %   df = dataloader(filename) reads the numeric dataset stored
    %   in a text/CSV file and returns it as a MATLAB table. The first
    %   five columns are interpreted as:
    %       1. Unit number
    %       2. Cycle (time in cycles)
    %       3. Operational Setting 1
    %       4. Operational Setting 2
    %       5. Operational Setting 3
    %   All remaining columns are treated as sensor measurements and
    %   automatically named Sensor1, Sensor2, ...
    %
    %   This function preserves rows with missing values, filling
    %   them with NaN.
    %
    %   Example:
    %       df = dataloader("engine_data.txt");
    %       head(df);

    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    data = {};
    while ~feof(fid)
        line = fgetl(fid);
        if isempty(line) || all(isspace(line))
            continue; % skip empty lines
        end
        % Split line by whitespace and convert to numbers
        nums = str2double(strsplit(strtrim(line)));
        data{end+1,1} = nums; %#ok<AGROW>
    end
    fclose(fid);

    % Determine maximum number of columns
    maxCols = max(cellfun(@numel, data));

    % Pad shorter rows with NaN
    for i = 1:numel(data)
        nPad = maxCols - numel(data{i});
        if nPad > 0
            data{i} = [data{i}, NaN(1,nPad)];
        end
    end

    % Convert to numeric matrix
    raw = vertcat(data{:});

    % Define column names
    nMeta = 5;
    nCols = size(raw,2);
    varNames = ["Unit","Cycle","OpSet1","OpSet2","OpSet3"];
    for i = nMeta+1:nCols
        varNames(i) = "Sensor" + (i-nMeta);
    end

    % Convert to table
    df = array2table(raw, "VariableNames", varNames);
end
