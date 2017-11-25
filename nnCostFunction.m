function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a_1 = [ones(m,1) X];
z_2 = a_1 * Theta1';
a_2 = [ones(size(a_1,1),1) sigmoid(z_2)];
a_3 = sigmoid(a_2 * Theta2');

J = zeros(m,1);
y_k = zeros(size(m,1),num_labels);

for i=1:m
    temp = (1:num_labels)';
    temp = temp == y(i);
    y_k(i,:) = temp';
    J(i) = (log(a_3(i,:)) * (-temp))  - (log(1.0 - a_3(i,:)) * (1.0 - temp));
end

J = sum(J)/m;
J = J + (sum(sum(Theta1(:, 2:size(Theta1,2)).^2)) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)))*lambda/(2*m);

s_3 = a_3 - y_k;
sd_2 = (s_3 * Theta2);
s_2 = sd_2(:,2:end) .* sigmoidGradient(z_2);

Theta1_grad = ((Theta1_grad + s_2' * a_1) + lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)])/m;
Theta2_grad = ((Theta2_grad + s_3' * a_2) + lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end)])/m;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end