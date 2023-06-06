function GraspFunc_WriteSFC(index)
    % 天线基本参数
    D = 110;
    F = 33;
    num = 128;
    [X, Y] = meshgrid(linspace(-D/2, D/2, num));

    A = 0.8*D/2;
    kk = 0.3;
    Window = 1 ./ (1 + exp(-2*kk*(A - sqrt(X.^2 + Y.^2)))); % 较宽的变形窗

    K = 0.025*ones(5);
    deform = rand(num, num) - 0.5;
    deform = deform .* Window;

    Loop = 10;
    for i = 1:Loop
        deform = conv2(deform, K, 'same');
    end
    deform = deform * 1;

    for i = 1:num
        for j = 1:num
            base = (X(i,j)^2 + Y(i,j)^2)/(4*F);
        
            % 生成数据格式
            Total(num*(i-1) + j, :) = [X(i,j) Y(i,j) base+deform(i,j)];
            deform11(num*(i-1) + j, :)= [X(i,j) Y(i,j) deform(i,j)];
        end
    end

    % 画图显示效果
%     total = Total(:, 3);
%     total = reshape(total, num, num);
    %mesh(X, Y, total);
    deform_11 = deform11(:, 3);
    deform_11 = reshape(deform_11, num, num);
    % sfc数据保存
    b='g';
    c = num^2;
    save(sprintf('Surface%05d.sfc',index), 'b', 'c', '-ascii')
    save(sprintf('Surface%05d.sfc',index), 'Total', '-ascii', '-append')
    % 生成标签
    label = deform_11 + 6*10^(-4);
    label = uint16(label*5*10^(7));
    imwrite(label, sprintf('NNlabel%05d.png',index));
end