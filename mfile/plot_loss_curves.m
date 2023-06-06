x = linspace(1,100,100);


filename1 = 'D:\SMC\JCST\data\loss\train_loss.csv';
train_loss = csvread(filename1, 1, 2);

filename2 = 'D:\SMC\JCST\data\loss\val_loss.csv';
val_loss = csvread(filename2, 1, 2);

filename3 = 'D:\SMC\JCST\data\loss\train_mae.csv';
train_mae = csvread(filename3, 1, 2);

filename4 = 'D:\SMC\JCST\data\loss\val_mae.csv';
val_mae = csvread(filename4, 1, 2);

filename5 = 'D:\SMC\JCST\data\loss\batch_loss.csv';
batch_loss = csvread(filename5, 1, 1);

filename6 = 'D:\SMC\JCST\data\loss\batch_mae.csv';
batch_mae = csvread(filename6, 1, 1);

figure(1)
plot(x(1:60),train_loss(1:60),'-r*','linewidth',3,'MarkerSize',14);
hold on;
plot(x(1:60),val_loss(1:60),'-g*','linewidth',3,'MarkerSize',14);
set(gca,'FontSize',24);
xlabel('Epoch','fontsize',24);ylabel('Loss','fontsize',24);
legend('Training loss','Validation loss','fontsize',24);
legend('boxoff');grid on;
box off;


figure(2)
plot(batch_loss(100:600,1),batch_loss(100:600,2),'-r*','linewidth',3,'MarkerSize',14);
hold on;
plot(batch_mae(100:600,1),batch_mae(100:600,2),'-g*','linewidth',3,'MarkerSize',14);
set(gca,'FontSize',24);
xlabel('Batch','fontsize',24);ylabel('Loss/MAE','fontsize',24);
legend('Batch loss','Batch mae','fontsize',24);
legend('boxoff');
box off;grid on;

