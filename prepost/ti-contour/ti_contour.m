%%
clear;
clc;
calmstd = load('offshore_detrend.mat').calmstd;
x1 = calmstd.u;
x2 = calmstd.u_stdv./calmstd.u;

%% scatter plot
figure1=figure('Position', [100, 100, 1120, 420]);
subplot(1,1,1);
scatter(x1, x2,10,'.');
% xlim([3,27])
xlabel('{\it u} (m/s)')
ylabel('ti')
title('Measurement')
box on;
set(gca,'FontSize',14,'FontName','Times New Roman')

%% Get the maximum and mimimum value
nbins = 36/2;
bin_center = zeros(nbins,1);
ti_min = zeros(nbins,1);
ti_max = zeros(nbins,1);
for k = 1:nbins
    bin1 = 2*(k -1);
    bin2 = 2*k;
    bin_center(k) = 0.5*(bin1+bin2);
    index = (x1>=bin1) & (x1<bin2);
    ti_bin = x2(index);
    ti_min(k) = min(ti_bin)/1.1;
    ti_max(k) = max(ti_bin)*1.1;
end


% X_c = cell(1,10);
% for k = 1:10
%     X_c{k}= random(gmModel, 1e8);
% end
% nbins = 36/2;
% bin_center = zeros(nbins,1);
% ti_min = zeros(nbins,1);
% ti_max = zeros(nbins,1);
% for k = 1:nbins
%     bin1 = 2*(k -1);
%     bin2 = 2*k;
%     bin_center(k) = 0.5*(bin1+bin2);
%     t_min_all = zeros(1,10);
%     t_max_all = zeros(1,10);
%     for j = 1:10
%         x = X_c{j};
%         x1 = x(:,1);
%         x2 = x(:,2);
%         index = (x1>=bin1) & (x1<bin2);
%         ti_bin = x2(index);
%         t_min_all(j) = min(ti_bin);
%         t_max_all(j) = max(ti_bin);
%     end
%     ti_min(k) = max(0,min(t_min_all));
%     ti_max(k) = max(t_max_all);
% end
% 


%% plot 
hold on 
plot(bin_center, ti_min, DisplayName="countour\_min")
plot(bin_center, ti_max, DisplayName="countour\_max")
xlim([4,25])
legend()