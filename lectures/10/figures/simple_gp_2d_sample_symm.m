% Simple demo of drawing from a 2-dimensional GP
%
% David Duvenaud
% Jan 2012
% -=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-

function simple_gp_2d_sample_symm

rng(2)

N_1d = 30;
range = linspace( -5, 5, N_1d);   % Choose a set of x locations.
[x1, x2] = meshgrid( range);
x = [x1(:) x2(:)];
N = size(x, 1);
mu = zeros(N, 1);   % Set the mean of the GP to zero everywhere.

% Specify the covariance between function values.
sigma = NaN(N, N);
for j = 1:N
    for k = 1:N
        xa = x(j,1);
        ya = x(j,2);
        xp = x(k,1);
        yp = x(k,2);
        sigma(j,k) = covariance( [xa ya], [xp, yp] ) ...
                   + covariance( [ya xa], [xp, yp]);% ...
                   %+ covariance( [xa ya], [yp, xp]) ...
                   %+ covariance( [ya xa], [yp, xp]);
    end
end

figure; imagesc(sigma);

f = mvnrnd( mu, sigma );           % Draw a sample from a multivariate Gaussian.
figure;  surf(x1, x2, reshape(f, N_1d, N_1d));   % Plot the draw.
end

% Squared-exp covariance function:
function c = covariance(x, y)
    d = x - y;
    sqdist = sum(d.^2);
    c = exp( - 0.5 * ( sqdist ) );
end
