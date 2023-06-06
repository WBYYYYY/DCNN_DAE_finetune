function y = Cal_diffy_2d(M, data, dx)
    len_M = length(M);
    M_index_off = (len_M + 1) / 2;
    M_half = (len_M - 1) / 2;
    
    y = conv2(M', data) / dx; y = y(M_index_off:end-M_index_off+1, :);
    for ii = 1:M_half
        y(ii, :) = y(M_index_off, :); y(end-ii+1, :) = y(end-M_half, :);
    end
end