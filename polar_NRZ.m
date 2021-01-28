function [t, X, T_s] = polar_NRZ(bit_stream, bit_rate, E_b, sample_per_pulse)
%% POLAR_NRZ Polar NRZ Line Coding Function
% This function converts a bit stream as input with specefic bit rate and energy to Polar NRZ format
% 
% Parameters:
%% 
% # _*bit_stream*_: input bit stream
% # _*bit_rate*_: input stream bit rate $\left(R_b \right)$
% # _*E_b*_: input stream bit energy $\left(E_b \right)$
% # _*sample_per_pulse*_: number of samples at output for each pulse
% Outputs:
%% 
% # _*t*_: time axis vector
% # _*X*_: amplitude vector
% # _*T_s*_: sampling period
%% 
% Calculating total output signal duration in seconds
T = length(bit_stream) / bit_rate;
%% 
% Number of total samples at output
N = sample_per_pulse * length(bit_stream);
%% 
% Calculating magnitude amplitude of each pulse at output
% 
% $$E_b =A^2 T_b =\frac{A^2 }{R_b }\;\Longrightarrow \;A=\sqrt{{E_b \;R}_b }$$
A = sqrt(E_b * bit_rate);
t = linspace(0,T,N);
T_s = t(2) - t(1);
X(N) = 0;
%% 
% Calculating output signal
for i = 0:length(bit_stream) - 1
    if bit_stream(i + 1) == 1
        X(i * sample_per_pulse + 1 :  ( i+1 )*sample_per_pulse ) = A;
    else
       X(i * sample_per_pulse + 1 :  ( i+1 )*sample_per_pulse ) = -A; 
    end
end
end