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
    Eap = a1 + 1i*a2; % �����һ�к͵ڶ��зֱ��ǵ糡x�����ʵ�����鲿������y�����z�������ֵ̫С�ˣ�������Բ���
    Eap = abs(Eap);
    Eap_de = X_Re + 1i*X_Im;
    Eap_de = abs(Eap_de);
    
    Eap = reshape(Eap,num,num);   % ������תΪ��ά����
    Eap = Eap';                 % ����grasp��������û�б����Ǵ�x��ʼ����y�Ὺʼ���Ǵ�������ʼ���ǴӸ�����ʼ������Ҫ���������
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