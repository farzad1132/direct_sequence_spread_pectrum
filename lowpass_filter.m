function [h] = lowpass_filter(f, R_b, T_s)
%% LOWPASS_FILTER LowPass Filter
% This function implements a low pass filter
% 
% _*low pass filter specification:*_
%% 
% * _pass band frequency_: $f_s =2\;R_b$
% * _pass band ripple_: $R_P =0\;\textrm{db}$
% * _stop band frequency_: $f_p =2\;R_b$
%% 
% |_*Parameters:*_|
%% 
% # _*f*_: frequency vector
% # _*R_b*_: bit rate of signal
% # _*T_s*_: sampling period
%% 
% |_*Output:*_|
%% 
% # _*h*_: frequency responce of the filter
h(length(f)) = 0;
low_index = find(f < 2*R_b);
high_index = find(f > ((1/T_s) - 2*R_b));
index = [low_index high_index];
h(index) = 1;
end