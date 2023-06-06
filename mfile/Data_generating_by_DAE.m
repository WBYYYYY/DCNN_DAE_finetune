clear; clc;
D = 110; F = 33; h = 33;headerlines = 11;
num = 128;
[X, Y] = meshgrid(linspace(-D/2, D/2, num));
img_num = 20000;

for ii = 0:img_num-1
    ii

% ����һ������������ı���
KK = 0.025*ones(5);
deform = rand(num, num)-0.5;
A = 0.8*D/2;
kk = 0.3;
Window = 1 ./ (1 + exp(-2*kk*(A - sqrt(X.^2 + Y.^2)))); % �Ͽ�ı��δ�
deform = deform .* Window;
Loop = 10;
for i = 1:Loop
    deform = conv2(deform, KK, 'same');
end
deform(X.^2 + Y.^2 > (D)^2/4) = 0;

label = deform + 6*10^(-4);
label = uint16(label*5*10^(7));

% figure(1)
% surf(X,Y,deform);
% ������������δ���ΪTIFͼ��
imwrite(label, ['D:\unet\data\train_T\',num2str(ii),'.png']); 

% ���εĵ���
M = [1 0 -1]/2;
dx = X(1,2) - X(1,1);       % ��ɢ��ļ��
deformx = Cal_diffx_2d(M, deform, dx);
deformy = Cal_diffy_2d(M, deform, dx);

% ����G��H��R1��R2
G = 2*F.*(X.^2 + Y.^2 - 4*F*h)./((X.^2 + Y.^2) + 4*F^2+eps);
P = 2 ./ (X^2 + Y.^2 + 4*F^2);
U = -G .* P .* X; 
V = -G .* P .* Y;
Ux = Cal_diffx_2d(M, U, dx);
Vy = Cal_diffy_2d(M, V, dx);
H = Ux + Vy;
Gx = Cal_diffx_2d(M, G, dx);
Gy = Cal_diffy_2d(M, G, dx);
R1 = U + Gx;
R2 = V + Gy;

% ������׼ȷ��K
K1 = 2*F + (X.^2 + Y.^2)/(2*F) + 2*(X.*deformx + Y.*deformy) - 2*deform;
K2 = 1 + (X.^2 + Y.^2)/(4*F^2) + (X.*deformx + Y.*deformy)/F + (deformx.^2 + deformy.^2);
K = K1./K2;

% ������L
z = (X.^2 + Y.^2)/(4*F) + deform;
LL = (h - z) ./ (z - F + K);

% ����ͶӰ��
zx = X/(2*F) + deformx;
zy = Y/(2*F) + deformy;
x_star = X + LL .* (X - K .* zx);
y_star = Y + LL .* (Y - K .* zy);

% ���㵼��
x_star_x = Cal_diffx_2d(M, x_star, dx);
y_star_y = Cal_diffy_2d(M, y_star, dx);

% ��������FA
accurate_result = x_star_x .* y_star_y - 1;
accurate_result(X.^2 + Y.^2 > (D)^2/4) = 0;
train = accurate_result + (4*10^(-3));
train = uint16(train * (9*10^(6)));

imwrite(train,['D:\unet\data\train_H\',num2str(ii),'.png']);
end






