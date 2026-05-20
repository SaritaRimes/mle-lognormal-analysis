%% Implementacao do MLE Lognormal golden section %%

clear
close all
clc

% Definindo algumas constantes
mPu = 50;
snr = 2;
nBins = 100;
larguraBins = 5;
ocupacao = [10 30 50 80];
quantidadeAmostrasPositivas = zeros(4, 1);
numeroIteracoes = 10;
quantidadeSinais = 2000000;

% Definindo algumas estruturas
erros = cell(4, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes
amplitudes = cell(7, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes
divergenciakl = zeros(4, 4); % nas linhas os metodos e nas colunas as ocupacoes
estatisticaks = zeros(4, 4); % nas linhas os metodos e nas colunas as ocupacoes

divergenciaklGaussiano = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
divergenciaklOF2 = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
divergenciaklCOF = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
divergenciaklLognormal = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
estatisticaksGaussiano = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
estatisticaksOF2 = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
estatisticaksCOF = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes
estatisticaksLognormal = zeros(4, numeroIteracoes); % nas linhas as ocupacoes e nas colunas as iteracoes

probabilities = cell(5, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes
chi2 = cell(4, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes
stdErrorCutsProb = cell(4, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes
stdErrorCutsChi2 = cell(4, 4, numeroIteracoes); % nas linhas os metodos e nas colunas as ocupacoes

addpath('dataQuality', 'datasets', 'externalMethods', 'estimationMethods', 'efficiencyMethods');

% Pulso de referencia do calorimetro
s = [0 .0172 .4524 1 .5633 .1493 .0424];
% Metodo OF2
OF2 = [ -0.3781   -0.3572    0.1808    0.8125    0.2767   -0.2056   -0.3292 ];

indiceOcupacao = 1;
for oc = ocupacao
     
    % Carregando o ruido e definindo o pedestal
    ruidoTotal = load(['ruido_media' int2str(mPu) '/ruido_ocup' int2str(oc) '_' ...
                       int2str(quantidadeSinais) 'sinais.txt']);
    ped = 50;

    % Calculando o numero de sinais em cada um dos conjuntos
    numeroSinais = quantidadeSinais / numeroIteracoes;
    
    for it = 1:numeroIteracoes

        % Separando uma parte do conjunto de ruido
        inicio = (it - 1) * numeroSinais + 1;
        final = it * numeroSinais;
        ruido = ruidoTotal(inicio:final, :);
        
        % Dividindo o conjunto em ruido de treino e teste
        div = cvpartition(size(ruido, 1), 'HoldOut', 0.50);
        ind = div.test;
        ruidoTreino = ruido(ind,:);
        ruidoTeste = ruido(~ind,:);
        numeroEventos = size(ruidoTeste, 1);
        
        % Retirando sinais com amostras negativas
        indicesRuidoPositivo = -1*ones(size(ruidoTreino, 1), 1);
        for i = 1:size(ruidoTreino, 1)
            if ~any(ruidoTreino(i,:) <= 0)
                indicesRuidoPositivo(i) = i;
            end
        end
        indicesRuidoPositivo = indicesRuidoPositivo(indicesRuidoPositivo > 0);
        ruidoTreinoPositivo = ruidoTreino(indicesRuidoPositivo, :);
        
        % Parametros do MLE
        mediaLognormal = mean(log(ruidoTreinoPositivo));
        mediaGaussiana = mean(ruidoTreino);
        covarianciaLognormal = cov(log(ruidoTreinoPositivo));
        covarianciaGaussiana = cov(ruidoTreino);
        
        % Montando o sinal completo
        amplitudeVerdadeira = exprnd(snr*mPu, numeroEventos, 1);
        r = ones(size(ruidoTeste, 1), size(ruidoTeste, 2));
        for i = 1:numeroEventos
            r(i,:) = amplitudeVerdadeira(i)*s + ruidoTeste(i,:);
        end
        
        % Metodos Lineares
        OF = (covarianciaGaussiana\s')/((s/covarianciaGaussiana)*s');
        OF2 = tile_of2(ruidoTreino, 1);
        amplitudeGaussiano = (r - ped)*OF;
        amplitudeOF2 = r*OF2';
        amplitudeCOF = aplicaCOF(r - ped, 4.5);
        amplitudesSinaisCOF = aplicaCOFAll(r - ped, 4.5);
        
        % MLE Lognormal
        amplitudeLognormal = ones(size(r, 1)  , 1);
        amplitudeLognormalSA = ones(size(r, 1)  , 1);
        for i = 1:numeroEventos 
          
            fprintf(['Processando evento ' int2str(i) '/' int2str(size(r, 1)) ...
                     ', Ocupação ' int2str(oc), ', iteração ' int2str(it) '\n']);

            % Verificando se ha amostras negativas no sinal recebido
            if any(r(i, :) <= 0)
                amplitudeLognormal(i) = amplitudeGaussiano(i);
                amplitudeLognormalSA(i) = amplitudeGaussiano(i);
                continue;
            end
    
            % Estimando a amplitude via razão áurea
            amplitudeLognormalSA(i) ...
                = golden_section(@(A)pdflognormal(mediaLognormal, ...
                                                  covarianciaLognormal, ...
                                                  r(i,:), s, A), ...
                                 0, 1023, 1);

            if amplitudeLognormalSA(i) == 0
                amplitudeLognormalSA(i) = amplitudeGaussiano(i);
            end
            
        end
        
        % Calculando os erros
        erroGaussiano = amplitudeGaussiano - amplitudeVerdadeira;
        erroOF2 = amplitudeOF2 - amplitudeVerdadeira;
        erroCOF = amplitudeCOF - amplitudeVerdadeira;
        erroLognormal = amplitudeLognormal - amplitudeVerdadeira;
        erroLognormalSA = amplitudeLognormalSA - amplitudeVerdadeira;
        
        % Salvando os erros
        erros{1, indiceOcupacao, it} = erroGaussiano;
        erros{2, indiceOcupacao, it} = erroOF2;
        erros{3, indiceOcupacao, it} = erroCOF;
        erros{4, indiceOcupacao, it} = erroLognormalSA;

        % Calculando as probabilidades, os chi2 e os desvios padrões para 
        % data quality
        [probabilities(:, indiceOcupacao, it), ...
         chi2(:, indiceOcupacao, it), ...
         stdErrorCutsProb(:, indiceOcupacao, it), ...
         stdErrorCutsChi2(:, indiceOcupacao, it)] ...
            = analysisDataQuality({amplitudeOF2, amplitudeGaussiano, ...
                                   amplitudesSinaisCOF, amplitudeLognormalSA, ...
                                   amplitudeVerdadeira}, r, ...
                                  mediaLognormal, mediaGaussiana, ...
                                  covarianciaLognormal, covarianciaGaussiana, ...
                                  ped, oc);

        % Armazenando as amplitudes
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'efficiency/gaussiano/ocup' int2str(oc) '/amplitude_gaussiano_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', amplitudeGaussiano);
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'efficiency/of/ocup' int2str(oc) '/amplitude_of_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', amplitudeOF2);
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'efficiency/cof/ocup' int2str(oc) '/amplitude_cof_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', amplitudeCOF);
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'efficiency/lognormal/ocup' int2str(oc) '/amplitude_lognormal_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', amplitudeLognormalSA);
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'efficiency/verdadeira/ocup' int2str(oc) '/amplitude_verdadeira_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.d \t %.13f\n', [find(~ind == true)'; amplitudeVerdadeira']);
        amplitudes{1, indiceOcupacao, it} = amplitudeGaussiano;
        amplitudes{2, indiceOcupacao, it} = amplitudeOF2;
        amplitudes{3, indiceOcupacao, it} = amplitudeCOF;
        amplitudes{4, indiceOcupacao, it} = amplitudeLognormalSA;
        amplitudes{5, indiceOcupacao, it} = amplitudeVerdadeira;

        % Armazenando os erros para data quality
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/erro_data_quality/erro_mle_gaussiano_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', erros{1, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/erro_data_quality/erro_of_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', erros{2, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/erro_data_quality/erro_cof_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', erros{3, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/erro_data_quality/erro_mle_lognormal_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', erros{4, indiceOcupacao, it});

        % Armazenando as probabilidades
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/probabilidade/prob_mle_gaussiano_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.35f\n', probabilities{1, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/probabilidade/prob_of_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.35f\n', probabilities{2, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/probabilidade/prob_cof_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.35f\n', probabilities{3, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/probabilidade/prob_mle_lognormal_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.35f\n', probabilities{4, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/probabilidade/prob_mle_lognormalgauss_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.35f\n', probabilities{5, indiceOcupacao, it});

        % Armazenando os chi2
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/chi2/chi2_mle_gaussiano_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', chi2{1, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/chi2/chi2_of_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', chi2{2, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/chi2/chi2_cof_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', chi2{3, indiceOcupacao, it});
        path = ['D:/Documentos/UERJ/Doutorado/ArtigoIEEE/dados_txt/cross_validation/' ...
                'data_quality/ocup' int2str(oc) '/chi2/chi2_mle_lognormal_ocup' ...
                int2str(oc) '_it' int2str(it) '.txt'];
        fprintf(fopen(path, 'w'), '%.13f\n', chi2{4, indiceOcupacao, it});

        % Plotando os histogramas dos erros
        figure('Position', [50 100 900 600])
        nBins = round((max(erroGaussiano) - min(erroGaussiano)) / larguraBins);
        histogram(erroGaussiano, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 1.5);
        hold on
        nBins = round((max(erroOF2) - min(erroOF2)) / larguraBins);
        histogram(erroOF2, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'g', 'LineWidth', 1.5);
        hold on
        nBins = round((max(erroCOF) - min(erroCOF)) / larguraBins);
        histogram(erroCOF, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'k', 'LineWidth', 1.5);
        hold on
        nBins = round((max(erroLognormalSA) - min(erroLognormalSA)) / larguraBins);
        histogram(erroLognormalSA, nBins, 'DisplayStyle', 'stairs', 'EdgeColor', 'r', 'LineWidth', 1.5);
        hold off
        xlim([-300 600]);
        legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, ...
               'FontSize', 12, 'Location', 'northeast');
        title(['Histograma do erro padrão com ocupação ' int2str(oc) '%'], 'FontSize', 13);
        xlabel('Erro (contagens de ADC)');
        ylabel('Número de eventos');

    end

    indiceOcupacao = indiceOcupacao + 1;
end

% Plotando a quantidade de amostras positivas
figure('Position', [50 100 900 600]);
plot(ocupacao, quantidadeAmostrasPositivas, '.-', 'MarkerSize', 20);
xlim([0 100]);
ylim([0 120]);
xlabel('Ocupação (%)');
ylabel('Amostras positivas (%)');

% Plotando a divergência KL nas ocupacoes
figure('Position', [50 100 900 600]);
plot(ocupacao, divergenciakl(1, :), '.--', 'MarkerSize', 20, 'Color', 'b');
hold on
plot(ocupacao, divergenciakl(2, :), '.--', 'MarkerSize', 20, 'Color', 'g');
hold on
plot(ocupacao, divergenciakl(3, :), '.--', 'MarkerSize', 20, 'Color', 'r');
hold on
plot(ocupacao, divergenciakl(4, :), '.--', 'MarkerSize', 20, 'Color', 'm');
legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, 'FontSize', 15, 'Location', 'northwest');
xlabel('Ocupação (%)', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);

% Plotando a estatistica KS nas ocupacoes
figure('Position', [50 100 900 600]);
plot(ocupacao, estatisticaks(1, :), '.--', 'MarkerSize', 20, 'Color', 'b');
hold on
plot(ocupacao, estatisticaks(2, :), '.--', 'MarkerSize', 20, 'Color', 'g');
hold on
plot(ocupacao, estatisticaks(3, :), '.--', 'MarkerSize', 20, 'Color', 'r');
hold on
plot(ocupacao, estatisticaks(4, :), '.--', 'MarkerSize', 20, 'Color', 'm');
legend({'MLE Gaussiano', 'OF', 'COF', 'MLE Lognormal'}, 'FontSize', 15, 'Location', 'northwest');
xlabel('Ocupação (%)', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);

% Plotando a estatistica KS nas iteracoes
xIteracoes = 1:numeroIteracoes;
%%% MLE Gaussiano
figure('Position', [50 100 900 600]);
plot(xIteracoes, estatisticaksGaussiano(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksGaussiano(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksGaussiano(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, estatisticaksGaussiano(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(estatisticaksGaussiano)) - 0.01) (max(max(estatisticaksGaussiano)) + 0.01)]);
title('MLE Gaussiano', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% OF2
figure('Position', [50 100 900 600]);
plot(xIteracoes, estatisticaksOF2(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksOF2(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksOF2(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, estatisticaksOF2(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(estatisticaksOF2)) - 0.01) (max(max(estatisticaksOF2)) + 0.01)]);
title('OF2', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% COF
figure('Position', [50 100 900 600]);
plot(xIteracoes, estatisticaksCOF(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksCOF(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksCOF(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, estatisticaksCOF(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(estatisticaksCOF)) - 0.01) (max(max(estatisticaksCOF)) + 0.01)]);
title('COF', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);
%%% MLE Lognormal
figure('Position', [50 100 900 600]);
plot(xIteracoes, estatisticaksLognormal(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksLognormal(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, estatisticaksLognormal(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, estatisticaksLognormal(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(estatisticaksLognormal)) - 0.005) (max(max(estatisticaksLognormal)) + 0.005)]);
title('MLE Lognormal', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Estatística KS', 'FontSize', 15);

% Plotando a divergencia KL nas iteracoes
%%% MLE Gaussiano
figure('Position', [50 100 900 600]);
plot(xIteracoes, divergenciaklGaussiano(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklGaussiano(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklGaussiano(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, divergenciaklGaussiano(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(divergenciaklGaussiano)) - 0.01) (max(max(divergenciaklGaussiano)) + 0.01)]);
title('MLE Gaussiano', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% OF2
figure('Position', [50 100 900 600]);
plot(xIteracoes, divergenciaklOF2(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklOF2(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklOF2(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, divergenciaklOF2(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(divergenciaklOF2)) - 0.01) (max(max(divergenciaklOF2)) + 0.01)]);
title('OF2', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% COF
figure('Position', [50 100 900 600]);
plot(xIteracoes, divergenciaklCOF(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklCOF(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklCOF(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, divergenciaklCOF(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(divergenciaklCOF)) - 0.01) (max(max(divergenciaklCOF)) + 0.01)]);
title('COF', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);
%%% MLE Lognormal
figure('Position', [50 100 900 600]);
plot(xIteracoes, divergenciaklLognormal(1, :), '.--', 'Color', 'b', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklLognormal(2, :), '.--', 'Color', 'r', 'MarkerSize', 20);
hold on
plot(xIteracoes, divergenciaklLognormal(3, :), '.--', 'Color', 'g', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
hold on
plot(xIteracoes, divergenciaklLognormal(4, :), '.--', 'Color', 'm', 'MarkerSize', 20, 'MarkerFaceColor', 'b');
legend({'10%', '30%', '50%', '80%'}, 'FontSize', 15, 'Location', 'best');
xlim([0 11]);
ylim([(min(min(divergenciaklLognormal)) - 0.005) (max(max(divergenciaklLognormal)) + 0.005)]);
title('MLE Lognormal', 'FontSize', 15);
xlabel('Iterações', 'FontSize', 15);
ylabel('Divergência KL', 'FontSize', 15);






