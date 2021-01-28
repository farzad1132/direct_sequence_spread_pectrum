function [I, Q, I_symbol, Q_symbol] = ser2par(input , sample_per_pulse)
%% SER2PAR Serial To Parallel Function
% This function converts a serial signal with bit rate of R to two parallel signal with bit rate of $\frac{R_b }{2}$
% 
% 
% |_*Parameters*_|:
%% 
% # _*input*_: input bit stream
% # _*sample_per_pulse*_: number of samples at output for each pulse
%% 
% |_*Output*_|:
%% 
% # _*I:*_ I-Channel signal
% # _*Q:*_ Q-Channel signal
% # _*I_symbol:*_ I-channel symbol vector
% # _*Q_symbol:*_ Q-Channel symbol vector
n = 2 * sample_per_pulse;
I(length(input)) = 0;
Q(length(input)) = 0;
I_symbol(length(input)/(2*sample_per_pulse)) = 0;
Q_symbol(length(input)/(2*sample_per_pulse)) = 0;
k = 0;
X1_k = 0;
X2_k = 0;
for i = 1 : sample_per_pulse : length(input)
    if rem(k,2) == 0
        
        I(X1_k*n + 1 : (X1_k+1)*n) = pulse_generator(sign(input(i)), abs(input(i)), sample_per_pulse);
        I_symbol(X1_k+1) = input(i);
        X1_k = X1_k + 1;
    else
        amplitude = pulse_generator(sign(input(i)), abs(input(i)), sample_per_pulse);
        Q(X2_k*n + 1 : (X2_k+1)*n) = amplitude;
        Q_symbol(X2_k+1) = input(i);
        X2_k = X2_k + 1;
    end
    k = k + 1;
end
    function [x] = pulse_generator(sign, amplitude, sample_per_pulse)
        x(2 * sample_per_pulse) = 0;
        if sign > 0
            x = amplitude;
        else
            x = -amplitude;
        end
    end
end