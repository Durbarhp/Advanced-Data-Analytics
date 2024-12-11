%Data Set Chosen: Pavia Centre and university
data = load('Pavia.mat');
img = data.pavia; 

% structure of the data
[rows, cols, wavelengths] = size(img);

%RGB Viualization
hcube = hypercube('paviaU.dat','paviaU.hdr');
rgbImg = colorize(hcube, "Method", "rgb", "ContrastStretching", true);
figure
imshow(rgbImg)
title('RGB Image of pavia')
% Reference: https://se.mathworks.com/help/images/ref/hypercube.html#d126e349355

% Select bands for visualization 
rgb_bands = [102, 50, 30]; 
% I tried with the bands given in demo but it shows index must not exceed 102 
rgb_Img = img(:, :, rgb_bands);

% Normalize %Ref: https://se.mathworks.com/matlabcentral/answers/654113-how-can-i-normalize-a-greyscale-image
rgb_Img = mat2gray(rgb_Img);  

figure;
imshow(rgb_Img);
title('Visualization; Changing bands');


% Select bands for false color visualization
false_band = [30, 25, 10];  % I tried with different bands
false_colored = img(:, :, false_band);

% Normalize and display the false color image
false_colored = mat2gray(false_colored);  

figure;
imshow(false_colored);
title('False Color Visualization');

%Number of cells needed to store the data 

total_numCells = rows * cols * wavelengths;
disp(['Total number of cells: ', num2str(total_numCells)]);

% Reshape eshape the 3D matrix into a 2D matrix 

spectra2D = reshape(img, [], wavelengths);


% Display the size of the reshaped matrix
disp(['Size of reshaped matrix: ', num2str(size(spectra2D))]);

% Visualize a subset of reshaped matrix
subset = spectra2D(1:20, :);

% Display a subset of the reshaped matrix
figure;
imagesc(subset); 
colorbar;  
title('Subset of Reshaped Matrix');


%%%2

%%SVD

[U, S, V] = svd(spectra2D, "econ");

singularValues = diag(S);

figure; 
nexttile; 
bar(diag(S));
ylabel("Singular Value")
nexttile;
plot(cumsum(diag(S))/sum(diag(S)));
ylabel("Cummulative Singular Values");



% Plot cumulative explained variance
cummulative_variance = cumsum(singularValues.^2) / sum(singularValues.^2);

%Reference: https://stackoverflow.com/questions/23129563/understanding-of-cumsum-dfunction
figure;
plot(cummulative_variance, 'o-');
title('Cumulative Variance');
xlabel('Number of Singular Vectors');
ylabel('Cumulative Variance');


% Determine the reconstruction rank 
% Try to reconstruct 97%
Rank = find(cummulative_variance >= 0.97, 1);
disp(['Reconstruction Rank: ', num2str(Rank)]);


%Truncate the decomposed matrices according to the determined rank.

Trun_U = U(:, 1:Rank);
Trun_S = S(1:Rank, 1:Rank);
Trun_V = V(:, 1:Rank);



%ref:https://se.mathworks.com/matlabcentral/answers/453069-how-to-perform-a-truncated-svd



% . Given the left-hand singular matrix and the singular values diagonal matrices can be stored as (U*S)
%US = Trun_U .*Trun_V;
US = numel(Trun_U) * numel(Trun_V)
%number of elements in turncated U*V
ellements_US = numel(US);

%number of elements in turncated V
elements_V = numel(Trun_V);

% reduced dimensionality of the image after truncation
total_reduced_dimensionality = ellements_US + elements_V

% total number of elements in Pavia data
pavia_elements = rows * cols * wavelengths;

% Calculate percentage reduction in data dimension
reduced_percentage = 100 * (1 - total_reduced_dimensionality / pavia_elements);
disp(['Percentage of Dimension Reduction: ', num2str(reduced_percentage), '%']);

% Reconstruct the image
matrix_reconstruct = Trun_U *Trun_S * Trun_V';

% Reshape to the 3D again
reconstructed_Image = reshape(matrix_reconstruct, [rows, cols, wavelengths]);

% Calculate the RMSE between the new constructed image and the original
% image data
RMSE = sqrt(mean((img(:) - reconstructed_Image(:)).^2));
disp(['Reconstruction Error RMSE: ', num2str(RMSE)]);



%%%%%%%%%%%%EXTRA



% Reconstruction error
error_img = abs(img - reconstructed_Image);

% Arbritary bands for false color
err_img_F_color = error_img(:, :, [102, 70, 50]);  
err_img_F_color = mat2gray(err_img_F_color); 

% Display the false color image of the reconstruction error
figure;
imshow(err_img_F_color);
title('Showcase the false color image of the reconstruction error');
