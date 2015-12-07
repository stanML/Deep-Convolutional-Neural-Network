clear
clc

% Load training and test data

load('model_losses.mat')
load('predictions.mat')
load('targets.mat')

% Determine the size of the loss matrix
l_s = size(loss_characteristics);
loss = [];

% Concatenate loss values for each iteration

for i = 1:l_s(1)
    loss_iter = loss_characteristics(i,:);
    loss = horzcat(loss, loss_iter);

end

% Smooth loss function
loss_sm = smooth(loss, 20, 'lowess');

% Plot loss
subplot(2,1,1)
plot(loss_sm)
title('Loss Function')
xlabel('# Iterations')
ylabel('Cross-Entropy Error')

% Reshape predictions to a 2D matrix
% Examples are rows, tags are columns
predicts = reshape(predictions, 90,50);
c_preds = sum(predicts);
c_targs = sum(targets);

%Plot Cumalative Distribution
subplot(2,1,2)
x = linspace(1,50, 50);
plot(x, c_preds, 'r-o', x,c_targs, 'b-o')
legend('Predicted', 'Target')
title('Learnt Cumulative Distibution')
xlabel('Tags')
ylabel('Scores')

clear


