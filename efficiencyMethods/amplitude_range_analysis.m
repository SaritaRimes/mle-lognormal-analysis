function [meanError, stdError] = amplitude_range_analysis(amplitudesTrue, amplitudesEstimated, numberSets)
    numberSets = fix(numberSets);

    % Defining some constantes and structures
    amplitudesTrueRange = cell(numberSets, 1);
    amplitudesEstimatedRange = cell(numberSets, 1);
    meanError = zeros(numberSets, 1);
    stdError = zeros(numberSets, 1);

    % Checking some criteria
    if ~ismatrix(amplitudesTrue)
        throw('The amplitudes must be a matrix.');
    end

    amplitudesTrueIn = amplitudesTrue;
    amplitudesEstimatedIn = amplitudesEstimated;
    for i = 1:numberSets
        % Choosing the indexes of elements that satisfy the condition of range
        indexesIn = amplitudesTrueIn <= (i*(max(amplitudesTrue) / numberSets));

        % Storing the amplitudes in range
        amplitudesTrueRange{i, 1} = amplitudesTrueIn(indexesIn);
        amplitudesEstimatedRange{i, 1} = amplitudesEstimatedIn(indexesIn);

        % Estimating mean and standard deviation of error in range
        error = amplitudesEstimatedRange{i, 1} - amplitudesTrueRange{i, 1};
        if ~isempty(error)
            meanError(i) = mean(error);
            stdError(i) = std(error);
        else
            meanError(i) = 0;
            stdError(i) = 0;
        end

        indexesOut = ~indexesIn;
        amplitudesTrueIn = amplitudesTrueIn(indexesOut);
        amplitudesEstimatedIn = amplitudesEstimatedIn(indexesOut);
    end
end