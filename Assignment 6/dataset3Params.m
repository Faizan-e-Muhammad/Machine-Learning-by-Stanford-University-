function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


vals = [0.01 0.03 0.1 0.3 1 3 10 30];    % Array of values to test for C and sigma
tempC = 0;
tempSigma = 0;
errorMat = [];							 % Matrix to store C, sigma and error


% Two for loops to test all 64 values for C and sigma
for i = 1:8
	for j = 1:8
		tempC = vals(i);
		tempSigma = vals(j);

		% Training for specific values
		model= svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma)); 
		visualizeBoundary(X, y, model);

		% Predicting on cross validation set for specific values
		pred = svmPredict(model,Xval);
		predError = mean(double(pred ~= yval));		% Calculating error

		% Storing
		errorMat = [errorMat; tempC tempSigma predError];
	
	end
end

% Finding the minimum error 
[Y, I] = min(errorMat(:,3));

% C and sigma values that gave the minimum error
C = errorMat(I,1);
sigma = errorMat(I,2);

% =========================================================================

end
