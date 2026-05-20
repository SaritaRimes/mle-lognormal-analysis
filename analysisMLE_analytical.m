%implementacao do MLE lognormal força bruta

clear all
close all
clc

% Definindo algumas constantes
mPu = 50;
nBins = 100;
ocupacao = [10 30 50 80];
quantidadeAmostrasPositivas = zeros(4, 1);
numeroIteracoes = 10;

% Definindo algumas estruturas
erros = cell(4, 4); % nas linhas os metodos e nas colunas as ocupacoes
amplitudes = cell(5, 4); % nas linhas os metodos e nas colunas as ocupacoes
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


addpath('externalMethods');

for it = 1:numeroIteracoes
    indiceOcupacao = 1;
    for oc = ocupacao
    
        % Carregando o ruido e definindo o pedestal
        ruido = load(['D:/Documentos/UERJ/Doutorado/Simulacoes/RuidoSimuladoNovoSimulador/TileCal/ruido_media' ...
                      int2str(mPu) '/ruido_ocup' int2str(oc) '.txt']);
        ped = 50;
        
        % Retirando o pedestal do ruido
        ruido = ruido - ped;
        
        % Pulso de referencia do calorimetro
        s = [0 .0172 .4524 1 .5633 .1493 .0424];
        % Metodo OF2
        OF2 = [ -0.3781   -0.3572    0.1808    0.8125    0.2767   -0.2056   -0.3292 ];
        
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
        covarianciaLognormal = cov(log(ruidoTreinoPositivo));
        covarianciaGaussiana = cov(ruidoTreinoPositivo);
        
        % Montando o sinal completo
        %ampTrue = exprnd(mSinal,size(y,1),1);  
        amplitudeVerdadeira = 1023*rand(numeroEventos, 1);
        r = ones(size(ruidoTeste, 1), size(ruidoTeste, 2));
        for i = 1:numeroEventos
            r(i,:) = amplitudeVerdadeira(i)*s + ruidoTeste(i,:);
        end
        
        % Metodos Lineares
        OF = (inv(covarianciaGaussiana)*s')/(s*inv(covarianciaGaussiana)*s');
        amplitudeGaussiano = r*OF;
        amplitudeOF2 = r*OF2';
        amplitudeCOF = aplicaCOF(r, 4.5);
        
        % MLE Lognormal
        amplitudeLognormal = amplitudeGaussiano;
        amplitudeLognormalAnalitico = -10000*ones(numeroEventos, 1);
        constanteAnalitica = - exp(mediaLognormal' - (covarianciaLognormal * ones(7, 1)))';
        for i = 1:numeroEventos  
          
            fprintf(['Processando evento ' int2str(i) '/' int2str(numeroEventos) ...
                     ', Ocupação ' int2str(oc) '\n']);
        
            % Armazenando a probalidade para cada sinal (para debugar)
            prob = zeros(1,2047);
            probMax = 0;

            % Estimando a amplitude de forma analitica
            As = constanteAnalitica + r(i, :);
            amplitudeLognormalAnalitico(i) = As(4);
            
            % Computando Lognormal forca bruta
            cont = 1;
            for amplitudeAuxiliar = 0:0.5:1023
        
                ruidoAuxiliar = r(i,:) - amplitudeAuxiliar*s;
                if(min(ruidoAuxiliar) <= 0)
                    prob(cont) = 0;
                    continue;
                end
                
                prob(cont) = (1/(prod(ruidoAuxiliar)*sqrt(det(covarianciaLognormal))*(2*pi)^(3.5))) ...
                             *exp(-.5*((log(ruidoAuxiliar)-mediaLognormal) ...
                                  *inv(covarianciaLognormal) ...
                                  *(log(ruidoAuxiliar)-mediaLognormal)'));

                if prob(cont) > probMax
                    amplitudeLognormal(i) = amplitudeAuxiliar;
                    probMax = prob(cont);
                end
        
                cont = cont+1;
        
            end
        end    
        
        % Calculando os erros        
        erroGaussiano = amplitudeGaussiano - amplitudeVerdadeira;
        erroOF2 = amplitudeOF2 - amplitudeVerdadeira;
        erroCOF = amplitudeCOF - amplitudeVerdadeira;
        erroLognormal = amplitudeLognormal - amplitudeVerdadeira;
        
        % Armazenando os erros
        erros{1, indiceOcupacao} = erroGaussiano;
        erros{2, indiceOcupacao} = erroOF2;
        erros{3, indiceOcupacao} = erroCOF;
        erros{4, indiceOcupacao} = erroLognormal;
    
        % Armazenando as amplitudes
        amplitudes{1, indiceOcupacao} = amplitudeGaussiano;
        amplitudes{2, indiceOcupacao} = amplitudeOF2;
        amplitudes{3, indiceOcupacao} = amplitudeCOF;
        amplitudes{4, indiceOcupacao} = amplitudeLognormal;
        amplitudes{5, indiceOcupacao} = amplitudeVerdadeira;
    
        % Estimando a estatistica KS
        cdfGaussiano = histogram(amplitudeGaussiano, 100, 'Normalization', 'cdf').Values;
        cdfOF2 = histogram(amplitudeOF2, 100, 'Normalization', 'cdf').Values;
        cdfCOF = histogram(amplitudeCOF, 100, 'Normalization', 'cdf').Values;
        cdfLognormal = histogram(amplitudeLognormal, 100, 'Normalization', 'cdf').Values;
        cdfVerdadeiro = histogram(amplitudeVerdadeira, 100, 'Normalization', 'cdf').Values;
        [~,~, estatisticaks(1, indiceOcupacao)] = kstest2(cdfVerdadeiro, cdfGaussiano);
        [~,~, estatisticaks(2, indiceOcupacao)] = kstest2(cdfVerdadeiro, cdfOF2);
        [~,~, estatisticaks(3, indiceOcupacao)] = kstest2(cdfVerdadeiro, cdfCOF);
        [~,~, estatisticaks(4, indiceOcupacao)] = kstest2(cdfVerdadeiro, cdfLognormal);
    
        % Estimando a divergencia KL
        divergenciakl(1, indiceOcupacao) = calculatekldivergence(amplitudeGaussiano, amplitudeVerdadeira, 50, 5);
        divergenciakl(2, indiceOcupacao) = calculatekldivergence(amplitudeOF2, amplitudeVerdadeira, 50, 5);
        divergenciakl(3, indiceOcupacao) = calculatekldivergence(amplitudeCOF, amplitudeVerdadeira, 50, 5);
        divergenciakl(4, indiceOcupacao) = calculatekldivergence(amplitudeLognormal, amplitudeVerdadeira, 50, 5);

        % Armazenando a estatistica KS e a divergencia KL nas iteracoes
        estatisticaksGaussiano(indiceOcupacao, it) = estatisticaks(1, indiceOcupacao);
        estatisticaksOF2(indiceOcupacao, it) = estatisticaks(2, indiceOcupacao);
        estatisticaksCOF(indiceOcupacao, it) = estatisticaks(3, indiceOcupacao);
        estatisticaksLognormal(indiceOcupacao, it) = estatisticaks(4, indiceOcupacao);
        divergenciaklGaussiano(indiceOcupacao, it) = divergenciakl(1, indiceOcupacao);
        divergenciaklOF2(indiceOcupacao, it) = divergenciakl(2, indiceOcupacao);
        divergenciaklCOF(indiceOcupacao, it) = divergenciakl(3, indiceOcupacao);
        divergenciaklLognormal(indiceOcupacao, it) = divergenciakl(4, indiceOcupacao);
        
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
    
        indiceOcupacao = indiceOcupacao + 1;
    
    end
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






