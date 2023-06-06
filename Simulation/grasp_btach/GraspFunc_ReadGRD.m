function GraspFunc_ReadGRD(index)
    global sim_name;
    num = 128;D = 110;
    [X, Y] = meshgrid(linspace(-D/2, D/2, num));
    data = importdata([sim_name num2str(index) '.grd']);
    data0 = importdata('ideal_33m_128_0.3GHZ.grd');
    a1 = data0.data(end-num*num*6+1:6:end);
    a2 = data0.data(end-num*num*6+2:6:end);
    X_Re = data.data(end-num*num*6+1:6:end);
    X_Im = data.data(end-num*num*6+2:6:end);
    Eap = a1 + 1i*a2; % 这里第一列和第二列分别是电场x方向的实部和虚部，由于y方向和z方向的数值太小了，这里忽略不记
    Eap = abs(Eap);
    Eap_de = X_Re + 1i*X_Im;
    Eap_de = abs(Eap_de);
    
    Eap = reshape(Eap,num,num);   % 将数据转为二维矩阵
    Eap = Eap';                 % 由于grasp的输出结果没有标明是从x开始还是y轴开始，是从正方向开始还是从负方向开始，所以要做这个操作
    Eap_de = reshape(Eap_de,num,num); 
    Eap_de = Eap_de';
    E_noise = awgn(Eap_de,40,'measured');
    E_noise(X.^2 + Y.^2 > (D)^2/4) = 0;
    
    EA = Eap.^2 ./ (E_noise.^2);
    FA = EA - 1;
    FA(isnan(FA)) = 0;
    FA(isinf(FA)) = 0;
    FA(X.^2 + Y.^2 > (D)^2/4) = 0;
    train = FA + (4*10^(-3));
    train = uint16(train * (9*10^(6)));
    imwrite(train, sprintf('NNinput%05d.png',index), 'bitdepth', 16);
end