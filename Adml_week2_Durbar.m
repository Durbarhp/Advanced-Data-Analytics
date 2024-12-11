clc; clear all; close all; 

%%Let's load the data

data = readtable("water_potability.csv"); %CSV file 
columns = data.Properties.VariableNames; %columns of the data
data_array = table2array(data);
data_array(isnan(data_array)) = 0; % returns a logical array containing  
data_array = double(data_array); %Convert to double for computation

% Task 1
%Let's visualize the data set
figure;
boxplot(data_array, 'Labels', columns);
xlabel('Variables');ylabel('Values');
title('Water Potability data primary visualization');

%Task 2
%Scale and center the dataset

data_std = zscore(data_array); % standardize the data using z-score function

%%% Reference: https://se.mathworks.com/help/stats/zscore.html

%Revisualize the data after scaling and centering (standardization)
figure;
boxplot(data_std, 'Labels', columns);
xlabel('Variables');ylabel('Values');
title('Water potability data revisualized after standardizing');

% Task 3 
%Apply PCA on the model. 

[Loadings, Scores, Eigen_Values, T2,Explained, mu] = pca(data_std); %All variables are treated as independent inputs

%Task 4
%Visualize and comment the variation explained by the model with different no. PCs. 
D = diag(Eigen_Values);  %Diagonal matrix from Eigen Values

%ref: https://se.mathworks.com/matlabcentral/answers/73298-what-does-eigenvalues-expres-in-the-covariance-matrix
%Let's compute Cumulative sum of the singular values
Cum_D = cumsum(D)/sum(D);

%Visulaize with Stairstep graph
figure;
stairs(Cum_D); 

xlabel('Principal components');
ylabel('Cummulative variance explained')
title('Stairstep graph respresentation of principal component');


% 80% of the explained variance
 idx = find(cumsum(Explained)>80,1);

% Task 5
%Compute the biplot of the first two principal components and explain covariances on it
% Comment the biplot.
figure;
biplot(Loadings(:,1:2),'Scores',Scores(:,1:2),'VarLabels',columns);

% Task 6
%Using biplots, explain co-variations to the quality indicator
xlabel('PCA');
ylabel('Values')
title('PCA of the standardized Data');




% Task 7
%using loading bar plots, explain the importance of variables in the first principal components, and their variational profiles.
figure;
bar(Loadings(:,1));
title('Bar Plots of the first Principal Components and variational profile');

%%Task 8
%Plot the T2 
figure;
a = 1:length(T2);
plot(a,T2); 
title('T2 plot for the model');

%T2 with 2, 3 standard deviation from mean


T2_std_2 = mean(T2) + 2 * std(T2);
T2_std3 = mean(T2) + 3 * std(T2);

% T² Control Chart with 2, 3 standard deviation from the mean
figure;
a = 1:length(T2);
plot(a, T2, 'b-', 'LineWidth', 2); 
hold on;
yline(T2_std_2, 'r--', 'LineWidth', 2); % 2 standard deviation
yline(T2_std3, 'g--', 'LineWidth', 2); % 3 standard deviation
xlabel('Observations'); ylabel('T² Value');
title('T² Control Chart');
legend('T² values', '2 std', '3 std');
hold off;


% SPEX

% Calculate Squared prediction error
residuals = (data_std - Scores * Loadings');
spex = sum(residuals.^2, 2);

%% Reference: https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/interpreting-the-residuals
%%Reference: https://stats.stackexchange.com/questions/259343/squared-prediction-error-and-pca-derivation

% SPEx control chart with 2,3 standard deviation
figure;

spex_std_2 = mean(spex) + 2 * std(spex);
spex_std_3 = mean(spex) + 3 * std(spex);

plot(a, spex, '-o'); hold on;
plot([min(a) max(a)], [spex_std_2 spex_std_2], '--r', 'LineWidth', 1); % 2 standard deviation
plot([min(a) max(a)], [spex_std_3 spex_std_3], '--g', 'LineWidth', 1); % 3 Standard deviation
title('SPEx Control Chart');
xlabel('Observation'); ylabel('SPEx');
legend('SPEx values', '2 STD from mean', '3 STD from mean');
hold off;
