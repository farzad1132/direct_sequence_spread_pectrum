function [bit_stream, Rx_I_symbols, Rx_Q_symbols] = par2ser(I, Q, sample_per_pulse, Amplitude)
%% PAR2SER Correlation Detector and Parallel to Serial Function
% This function first find the correlation of I and Q channel signals, then it compares this correlation value with signals space and selects one with minimum difference
% |_*Parameters*_|:
%% 
% # _*I*_: I-Channel signal
% # _*Q*_: Q-Channel signal
% # _*sample_per_pulse*_: number of samples per one pulse at output
% # _*Amplitude*_: amplitude of baseband signal
%% 
% 
% 
% |_*outputs*_|:
%% 
% # _*bit_stream*_: output bit stream
% # _*Rx_I_symbols:*_ I-Channel symbol vector
% # _*Rx_Q_symbols:*_ Q-Channel symbol vector
%% 
% 
len = length(I)/(sample_per_pulse);
bit_stream(len) = 0;
Rx_I_symbols(len/2) = 0;
Rx_Q_symbols(len/2) = 0;
k = 1;
for i = 1 : len/2
%% 
% calculating correlation in I-channel
    cor = sum(I(1 + 2*(i-1)*sample_per_pulse : 2*i*sample_per_pulse)) / (2*sample_per_pulse);
    if cor > 0
        Rx_I_symbols(i) = sqrt(cor);
    else
        Rx_I_symbols(i) = -imag(sqrt(cor));
    end
    
%% 
% comparing correlation value with signal space 
    if abs(cor - Amplitude)^2 < abs(cor + Amplitude)^2
        bit_stream(k) = 1;
    end
    
    k = k + 1;
%% 
% calculating correlation in Q-channel
    cor = sum(Q(1 + 2*(i-1)*sample_per_pulse : 2*i*sample_per_pulse)) / (2*sample_per_pulse);
    if cor > 0
        Rx_Q_symbols(i) = sqrt(cor);
    else
        Rx_Q_symbols(i) = -imag(sqrt(cor));
    end
%% 
% comparing correlation value with signal space 
    if abs(cor - Amplitude)^2 < abs(cor + Amplitude)^2
        bit_stream(k) = 1;
    end
    k = k + 1;
end
end