function kl_divergence = calculatekldivergence(distribution1, distribution2, initialBin, finalBin)

    % Find the first bin value that has no zeros for both distributions
    for bin = initialBin:-5:finalBin
        count1 = histcounts(distribution1, bin);
        count2 = histcounts(distribution2, bin);

        if (min(count1) ~= 0 && min(count2) ~= 0)
            % Normalize both distributions so that the sum is equal to 1
            count1 = count1 / sum(count1);
            count2 = count2 / sum(count2);
        
            % Estimate the Kullback-Leibler Divergence
            kl_divergence = sum(count1 .* log2(count1 ./ count2));
            break;
        end
    end

end