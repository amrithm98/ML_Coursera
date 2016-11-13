function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% You need to return the following variables correctly.
C =1;
sigma = 0.3;
cur_error=100;
c1=0.01;
s1=0.01;
count=0;
while(c1<30)
	s1=0.01;
	while(s1<30)
		s1*=3;
		model=svmTrain(X,y,c1,@(x1,x2)gaussianKernel(x1,x2,s1));
		predictions = svmPredict(model, Xval);
		error=mean(double(predictions ~= yval));
		if(error<cur_error)
			cur_error=error;
			C=c1;
			sigma=s1;
		endif;
		count=count+1;
		disp(count);% training number
	endwhile;
	c1*=3;
endwhile;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
