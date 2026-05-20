%implementacao do MLE lognormal força bruta

clear all
close all
clc

% Definindo algumas constantes
mPu = 50;
nBins = 100;
occupancies = [10 30 50 80];
quantity_positive_samples = zeros(4, 1);
number_iterations = 1;

% Definindo algumas estruturas
errors = cell(4, 4); % nas linhas os metodos e nas colunas as ocupacoes
amplitudes = cell(5, 4); % nas linhas os metodos e nas colunas as ocupacoes
kl_divergence = zeros(4, 4); % nas linhas os metodos e nas colunas as ocupacoes
ks_statistic = zeros(4, 4); % nas linhas os metodos e nas colunas as ocupacoes

kl_divergence_gauss = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
kl_divergence_of = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
kl_divergence_cof = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
kl_divergence_logn = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
ks_statistic_gauss = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
ks_statistic_of = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
ks_statistic_cof = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes
ks_statistic_logn = zeros(4, number_iterations); % nas linhas as ocupacoes e nas colunas as iteracoes


addpath('internalMethods', 'externalMethods', 'efficiencyMethods', 'FastICA_25');

for it = 1:number_iterations
    occupancy_index = 1;
    for oc = occupancies
    
        % Carregando o ruido e definindo o pedestal
        noise = load(['D:/Documentos/UERJ/Doutorado/Simulacoes/RuidoSimuladoNovoSimulador/TileCal/ruido_media' ...
                      int2str(mPu) '/ruido_ocup' int2str(oc) '.txt']);
        ped = 50;
        
        % Retirando o pedestal do ruido
        noise = noise - ped;
        
        % Pulso de referencia do calorimetro
        s = [0 .0172 .4524 1 .5633 .1493 .0424];
        % Metodo OF2
        OF2 = [ -0.3781   -0.3572    0.1808    0.8125    0.2767   -0.2056   -0.3292 ];
        
        % Retirando sinais com amostras negativas
        k = 1;
        for i = 1:size(noise, 1)
            if min(noise(i,:)) >= 0
                noise_positive(k,:) = noise(i,:);
                k = k + 1;
            end
        end
        quantity_positive_samples(occupancy_index) = (size(noise_positive, 1) / size(noise, 1)) * 100;
        
        % Dividindo o conjunto em ruido de treino e teste
        div = cvpartition(size(noise_positive, 1), 'HoldOut', 0.50);
        ind = div.test;
        noise_training = noise_positive(ind,:);
        noise_test = noise_positive(~ind,:);
        number_events = size(noise_test, 1);
        
        % Parametros do MLE
        mean_logn = mean(log(noise_training));
        covariance_logn = cov(log(noise_training));
        covariance_gauss = cov(noise_training);
        
        % Montando o sinal completo
        %ampTrue = exprnd(mSinal,size(y,1),1);  
        amplitude_true = 1023*rand(number_events, 1);
        r = ones(size(noise_test, 1), size(noise_test, 2));
        for i = 1:number_events
            r(i,:) = amplitude_true(i)*s + noise_test(i,:);
        end
        
        % Metodos Lineares
        OF = (inv(covariance_gauss)*s')/(s*inv(covariance_gauss)*s');
        amplitude_gauss = r*OF;
        amplitude_of = r*OF2';
        amplitude_cof = aplicaCOF(r, 4.5);
        
        % MLE Lognormal
        amplitude_logn = amplitude_gauss;    
        for i = 1:number_events  
          
            fprintf(['Processando evento ' int2str(i) '/' int2str(number_events) ...
                     ', Ocupação ' int2str(oc) '\n']);
        
            % Armazenando a probalidade para cada sinal (para debugar)
            prob = zeros(1,2047);
            prob_max = 0;
            
            % Computando Lognormal forca bruta
            cont = 1;
            for amplitude_auxiliary = 0:0.5:1023
        
                noise_auxiliary = r(i,:) - amplitude_auxiliary*s;
                if(min(noise_auxiliary) <= 0)
                    prob(cont) = 0;
                    continue;
                end
                
                prob(cont) = (1/(prod(noise_auxiliary)*sqrt(det(covariance_logn))*(2*pi)^(3.5))) ...
                             *exp(-.5*((log(noise_auxiliary)-mean_logn) ...
                                  *inv(covariance_logn) ...
                                  *(log(noise_auxiliary)-mean_logn)'));
                if prob(cont) > prob_max
                    amplitude_logn(i) = amplitude_auxiliary;
                    prob_max = prob(cont);
                end
        
                cont = cont+1;
        
            end
        end    
        
        % Calculando os erros        
        erroGaussiano = amplitude_gauss - amplitude_true;
        erroOF2 = amplitude_of - amplitude_true;
        erroCOF = amplitude_cof - amplitude_true;
        erroLognormal = amplitude_logn - amplitude_true;
        
        % Armazenando os erros
        errors{1, occupancy_index} = erroGaussiano;
        errors{2, occupancy_index} = erroOF2;
        errors{3, occupancy_index} = erroCOF;
        errors{4, occupancy_index} = erroLognormal;
    
        % Armazenando as amplitudes
        amplitudes{1, occupancy_index} = amplitude_gauss;
        amplitudes{2, occupancy_index} = amplitude_of;
        amplitudes{3, occupancy_index} = amplitude_cof;
        amplitudes{4, occupancy_index} = amplitude_logn;
        amplitudes{5, occupancy_index} = amplitude_true;
    
        % Estimando a estatistica KS
        cdfGaussiano = histogram(amplitude_gauss, 100, 'Normalization', 'cdf').Values;
        cdfOF2 = histogram(amplitude_of, 100, 'Normalization', 'cdf').Values;
        cdfCOF = histogram(amplitude_cof, 100, 'Normalization', 'cdf').Values;
        cdfLognormal = histogram(amplitude_logn, 100, 'Normalization', 'cdf').Values;
        cdfVerdadeiro = histogram(amplitude_true, 100, 'Normalization', 'cdf').Values;
        [~,~, ks_statistic(1, occupancy_index)] = kstest2(cdfVerdadeiro, cdfGaussiano);
        [~,~, ks_statistic(2, occupancy_index)] = kstest2(cdfVerdadeiro, cdfOF2);
        [~,~, ks_statistic(3, occupancy_index)] = kstest2(cdfVerdadeiro, cdfCOF);
        [~,~, ks_statistic(4, occupancy_index)] = kstest2(cdfVerdadeiro, cdfLognormal);
    
        % Estimando a divergencia KL
        kl_divergence(1, occupancy_index) = calculatekldivergence(amplitude_gauss, amplitude_true, 50, 5);
        kl_divergence(2, occupancy_index) = calculatekldivergence(amplitude_of, amplitude_true, 50, 5);
        kl_divergence(3, occupancy_index) = calculatekldivergence(amplitude_cof, amplitude_true, 50, 5);
        kl_divergence(4, occupancy_index) = calculatekldivergence(amplitude_logn, amplitude_true, 50, 5);

        % Armazenando a estatistica KS e a divergencia KL nas iteracoes
        ks_statistic_gauss(occupancy_index, it) = ks_statistic(1, occupancy_index);
        ks_statistic_of(occupancy_index, it) = ks_statistic(2, occupancy_index);
        ks_statistic_cof(occupancy_index, it) = ks_statistic(3, occupancy_index);
        ks_statistic_logn(occupancy_index, it) = ks_statistic(4, occupancy_index);
        kl_divergence_gauss(occupancy_index, it) = kl_divergence(1, occupancy_index);
        kl_divergence_of(occupancy_index, it) = kl_divergence(2, occupancy_index);
        kl_divergence_cof(occupancy_index, it) = kl_divergence(3, occupancy_index);
        kl_divergence_logn(occupancy_index, it) = kl_divergence(4, occupancy_index);
        
        % Plotando os histogramas dos erros
        figure('Position', [50 100 900 600])
        histogram(erroGaussiano, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 1.5);
        hold on
        histogram(erroOF2, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'g', 'LineWidth', 1.5);
        hold on
        histogram(erroCOF, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'k', 'LineWidth', 1.5);
        hold on
        histogram(erroLognormal, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'm', 'LineWidth', 1.5);    
        hold off
        xlim([-300 600]);
        legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, 'FontSize', 12, 'Location', 'northeast');
        title(['Histograma do erro padrão com ocupação ' int2str(oc) '%'], 'FontSize', 13);
        xlabel('Erro (contagens de ADC)');
        ylabel('Número de eventos');
    
        occupancy_index = occupancy_index + 1;
    
    end
end

% Plotando a quantidade de amostras positivas
figure('Position', [50 100 900 600]);
plot(occupancies, quantity_positive_samples, '.-', 'MarkerSize', 20);
xlim([0 100]);
ylim([0 120]);
xlabel('Ocupação (%)');
ylabel('Amostras positivas (%)');

% Plotando a divergência KL nas ocupacoes
figure('Position', [50 100 900 600]);
plot(occupancies, kl_divergence(1, :), '.--', 'MarkerSize', 20, 'Color', 'b');
hold on
plot(occupancies, kl_divergence(2, :), '.--', 'MarkerSize', 20, 'Color', 'g');
hold on
plot(occupancies, kl_divergence(3, :), '.--', 'MarkerSize', 20, 'Color', 'r');
hold on
plot(occupancies, kl_divergence(4, :), '.--', 'MarkerSize', 20, 'Color', 'm');
legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, 'FontSize', 15, 'Location', 'northwest');
xlabel('Ocupação (%)', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);

% Plotando a estatistica KS nas ocupacoes
figure('Position', [50 100 900 600]);
plot(occupancies, ks_statistic(1, :), '.--', 'MarkerSize', 20, 'Color', 'b');
hold on
plot(occupancies, ks_statistic(2, :), '.--', 'MarkerSize', 20, 'Color', 'g');
hold on
plot(occupancies, ks_statistic(3, :), '.--', 'MarkerSize', 20, 'Color', 'r');
hold on
plot(occupancies, ks_statistic(4, :), '.--', 'MarkerSize', 20, 'Color', 'm');
legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, 'FontSize', 15, 'Location', 'northwest');
xlabel('Ocupação (%)', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);

% Plotando a estatistica KS nas iteracoes
xIteracoes = 1:number_iterations;
%%% MLE Gaussiano
figure('Position', [50 100 900 600]);
plot(xIteracoes, ks_statistic_gauss(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_gauss(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_gauss(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, ks_statistic_gauss(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(ks_statistic_gauss)) - 0.01) (max(max(ks_statistic_gauss)) + 0.01)]);
title('MLE Gaussiano', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% OF2
figure('Position', [50 100 900 600]);
plot(xIteracoes, ks_statistic_of(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_of(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_of(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, ks_statistic_of(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(ks_statistic_of)) - 0.01) (max(max(ks_statistic_of)) + 0.01)]);
title('OF2', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% COF
figure('Position', [50 100 900 600]);
plot(xIteracoes, ks_statistic_cof(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_cof(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_cof(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, ks_statistic_cof(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(ks_statistic_cof)) - 0.01) (max(max(ks_statistic_cof)) + 0.01)]);
title('COF', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% MLE Lognormal
figure('Position', [50 100 900 600]);
plot(xIteracoes, ks_statistic_logn(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_logn(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, ks_statistic_logn(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, ks_statistic_logn(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(ks_statistic_logn)) - 0.005) (max(max(ks_statistic_logn)) + 0.005)]);
title('MLE Lognormal', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);

% Plotando a divergencia KL nas iteracoes
%%% MLE Gaussiano
figure('Position', [50 100 900 600]);
plot(xIteracoes, kl_divergence_gauss(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_gauss(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_gauss(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, kl_divergence_gauss(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(kl_divergence_gauss)) - 0.01) (max(max(kl_divergence_gauss)) + 0.01)]);
title('MLE Gaussiano', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% OF2
figure('Position', [50 100 900 600]);
plot(xIteracoes, kl_divergence_of(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_of(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_of(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, kl_divergence_of(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(kl_divergence_of)) - 0.01) (max(max(kl_divergence_of)) + 0.01)]);
title('OF2', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% COF
figure('Position', [50 100 900 600]);
plot(xIteracoes, kl_divergence_cof(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_cof(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_cof(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, kl_divergence_cof(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(kl_divergence_cof)) - 0.01) (max(max(kl_divergence_cof)) + 0.01)]);
title('COF', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% MLE Lognormal
figure('Position', [50 100 900 600]);
plot(xIteracoes, kl_divergence_logn(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_logn(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, kl_divergence_logn(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, kl_divergence_logn(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(kl_divergence_logn)) - 0.005) (max(max(kl_divergence_logn)) + 0.005)]);
title('MLE Lognormal', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);






