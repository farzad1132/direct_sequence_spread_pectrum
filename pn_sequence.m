function [x] = pn_sequence(polynomial, PG, symbol_per_pulse, data_bit_count, data_bit_rate)
%% PN_SEQUENCE PN Sequence Generator
% This method calculates PN sequece with Fibonacci format
% 
% Parameters:
%% 
% # _*polynomial*_: PN sequence generator polynomial in matrix form
% # _*PG*_: ratio of spreaded sginal bandwidth to original signal bandwidth
% # _*symbol_per_pulse*_: number of samples per one pulse at output
% # _*data_bit_count*_: length of input data bit stream that
% # _*data_bit_rate*_: input data bit rate
%% 
% _*NOTE 1*_: $\textrm{PG}=\frac{T_b }{T_c }$,    where $T_b$ is bit rate of 
% base band input data bit stream and $T_c$ is chip rate of PN sequence
% 
% _*NOTE 2*_: $\textrm{PG}$ must be an integer and $\textrm{PG}\ge 1\ldotp$ 
% also symbol per pulse number must be dividable by PG
if PG < 1 || rem(PG,1) ~= 0 || rem(symbol_per_pulse, PG) ~= 0
    error("PG must be an integer and greater than 1 ( or equal ) and\n" + ...
        " symbol per pulse number must be dividable by PG");
end
n = symbol_per_pulse / PG;
chip_rate = data_bit_rate / PG;
order = length(polynomial) - 1;
mask(order) = 1;
init(order) = 0;
init(1) = 1;
Seqcount = PG * data_bit_count;
pnSequence = comm.PNSequence('Polynomial',polynomial,'InitialConditions', init,...
'SamplesPerFrame',Seqcount, 'Mask', mask);
pn = pnSequence();
bit_stream = pn(1:Seqcount);
[~, x] = polar_NRZ(bit_stream, chip_rate, (1 / (chip_rate)), n);
end