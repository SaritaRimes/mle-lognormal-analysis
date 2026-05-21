%% Lognormal MLE Implementation using Golden-Section Search %%

clear
close all
clc

% Defining some constants parameters
mPu = 50;
snr = 2;
nBins = 100;
withBins = 5;
occupancy = [10 30 50 80];
positiveSamplesQuantity = zeros(4, 1);
nIterations = 10;
signalsQuantity = 2000000;

% Defining some structures
error = cell(4, 4, nIterations); % rows: methods, columns: occupancies
amplitudes = cell(7, 4, nIterations);
klDivergence = zeros(4, 4);
ksStatistic = zeros(4, 4);

klDivergenceGauss = zeros(4, nIterations); % rows: methods, columns: occupancies
klDivergenceOF2 = zeros(4, nIterations);
klDivergenceCOF = zeros(4, nIterations);
klDivergenceLogn = zeros(4, nIterations);
ksStatisticGauss = zeros(4, nIterations);
ksStatisticOF2 = zeros(4, nIterations);
ksStatisticCOF = zeros(4, nIterations);
ksStatisticLogn = zeros(4, nIterations);

probabilities = cell(5, 4, nIterations); % rows: methods, columns: occupancies
chi2 = cell(4, 4, nIterations);
stdErrorCutsProb = cell(4, 4, nIterations);
stdErrorCutsChi2 = cell(4, 4, nIterations);

addpath('dataQuality', 'datasets', 'externalMethods', 'estimationMethods', 'efficiencyMethods');

% Calorimeter reference pulse
s = [0 .0172 .4524 1 .5633 .1493 .0424];
% OF2 weights
OF2 = [ -0.3781   -0.3572    0.1808    0.8125    0.2767   -0.2056   -0.3292 ];

indexOccupancy = 1;
for oc = occupancy
     
    % Loading the noise and defining the pedestal
    totalNoise = load(['ruido_media' int2str(mPu) '/ruido_ocup' int2str(oc) '_' ...
                       int2str(signalsQuantity) 'sinais.txt']);
    ped = 50;

    % Calculating the number of signals in each dataset
    nSignals = signalsQuantity / nIterations;
    
    for it = 1:nIterations

        % Selecting the corresponding part of the noise dataset
        start = (it - 1) * nSignals + 1;
        final = it * nSignals;
        noise = totalNoise(start:final, :);
        
        % Splitting the dataset into training and test noise
        div = cvpartition(size(noise, 1), 'HoldOut', 0.50);
        ind = div.test;
        noiseTraining = noise(ind,:);
        noiseTest = noise(~ind,:);
        nEvents = size(noiseTest, 1);
        
        % Removing signals with negative samples
        indexPositiveNoise = -1*ones(size(noiseTraining, 1), 1);
        for i = 1:size(noiseTraining, 1)
            if ~any(noiseTraining(i,:) <= 0)
                indexPositiveNoise(i) = i;
            end
        end
        indexPositiveNoise = indexPositiveNoise(indexPositiveNoise > 0);
        noiseTrainingPositive = noiseTraining(indexPositiveNoise, :);
        
        % MLE parameters
        meanLogn = mean(log(noiseTrainingPositive));
        meanGauss = mean(noiseTraining);
        covLogn = cov(log(noiseTrainingPositive));
        covGauss = cov(noiseTraining);
        
        % Assembling the complete signal
        amplitudeTrue = exprnd(snr*mPu, nEvents, 1);
        r = ones(size(noiseTest, 1), size(noiseTest, 2));
        for i = 1:nEvents
            r(i,:) = amplitudeTrue(i)*s + noiseTest(i,:);
        end
        
        % Linear methods
        OF = (covGauss\s')/((s/covGauss)*s');
        OF2 = tile_of2(noiseTraining, 1);
        amplitudeGauss = (r - ped)*OF;
        amplitudeOF2 = r*OF2';
        amplitudeCOF = aplicaCOF(r - ped, 4.5);
        amplitudesSignalsCOF = aplicaCOFAll(r - ped, 4.5);
        
        % Lognormal MLE
        amplitudeLogn = ones(size(r, 1)  , 1);
        for i = 1:nEvents 
          
            fprintf(['Processando evento ' int2str(i) '/' int2str(size(r, 1)) ...
                     ', Ocupação ' int2str(oc), ', iteração ' int2str(it) '\n']);

            % Verificando se ha amostras negativas no sinal recebido
            if any(r(i, :) <= 0)
                amplitudeLogn(i) = amplitudeGauss(i);
                continue;
            end
    
            % Estimando a amplitude via razão áurea
            amplitudeLogn(i) ...
                = golden_section(@(A)pdflognormal(meanLogn, ...
                                                  covLogn, ...
                                                  r(i,:), s, A), ...
                                 0, 1023, 1);

            if amplitudeLogn(i) == 0
                amplitudeLogn(i) = amplitudeGauss(i);
            end
            
        end
        
        % Calculating the errors
        errorGauss = amplitudeGauss - amplitudeTrue;
        errorOF2 = amplitudeOF2 - amplitudeTrue;
        errorCOF = amplitudeCOF - amplitudeTrue;
        errorLogn = amplitudeLogn - amplitudeTrue;
        
        % Storing the errors
        error{1, indexOccupancy, it} = errorGauss;
        error{2, indexOccupancy, it} = errorOF2;
        error{3, indexOccupancy, it} = errorCOF;
        error{4, indexOccupancy, it} = errorLogn;

        % Calculating probability, chi2 and standard deviation for data quality
        [probabilities(:, indexOccupancy, it), ...
         chi2(:, indexOccupancy, it), ...
         stdErrorCutsProb(:, indexOccupancy, it), ...
         stdErrorCutsChi2(:, indexOccupancy, it)] ...
            = analysisDataQuality({amplitudeOF2, amplitudeGauss, ...
                                   amplitudesSignalsCOF, amplitudeLogn, ...
                                   amplitudeTrue}, r, ...
                                  meanLogn, meanGauss, ...
                                  covLogn, covGauss, ...
                                  ped, oc);

        % Storing the amplitudes
        mkdir('outputData/amplitudes');
        path = ['/outputData/amplitudes/amp_gauss_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', amplitudeGauss);
        fclose(f);
        %
        path = ['/outputData/amplitudes/amp_of_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', amplitudeOF2);
        fclose(f);
        %
        path = ['/outputData/amplitudes/amp_cof_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', amplitudeCOF);
        fclose(f);
        %
        path = ['/outputData/amplitudes/amp_logn_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', amplitudeLogn);
        fclose(f);
        %
        path = ['/outputData/amplitudes/amp_true_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.d \t %.13f\n', [find(~ind == true)'; amplitudeTrue']);
        fclose(f);
        %
        amplitudes{1, indexOccupancy, it} = amplitudeGauss;
        amplitudes{2, indexOccupancy, it} = amplitudeOF2;
        amplitudes{3, indexOccupancy, it} = amplitudeCOF;
        amplitudes{4, indexOccupancy, it} = amplitudeLogn;
        amplitudes{5, indexOccupancy, it} = amplitudeTrue;

        % Storing the errors for data quality
        mkdir('outputData/dataQuality');
        path = ['/outputData/dataQuality/dq_error_mle_gauss_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', error{1, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_error_of_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', error{2, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_error_cof_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', error{3, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_error_mle_logn_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', error{4, indexOccupancy, it});
        fclose(f);

        % Storing the probabilities for data quality
        mkdir('outputData/dataQuality');
        path = ['/outputData/dataQuality/dq_prob_mle_gauss_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.35f\n', probabilities{1, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_prob_of_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.35f\n', probabilities{2, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_prob_cof_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.35f\n', probabilities{3, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_prob_mle_logn_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.35f\n', probabilities{4, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_prob_mle_logn-gauss_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.35f\n', probabilities{5, indexOccupancy, it});
        fclose(f);

        % Storing the chi2 for data quality
        mkdir('outputData/dataQuality');
        path = ['/outputData/dataQuality/dq_chi2_mle_gauss_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', chi2{1, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_chi2_of_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', chi2{2, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_chi2_cof_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', chi2{3, indexOccupancy, it});
        fclose(f);
        %
        path = ['/outputData/dataQuality/dq_chi2_mle_logn_ocup' int2str(oc) '_it' int2str(it) '.txt'];
        f = fopen(path, 'w');
        fprintf(f, '%.13f\n', chi2{4, indexOccupancy, it});
        fclose(f);

    end

    indexOccupancy = indexOccupancy + 1;
end
