data_train = dlmread("train_T.csv", ",", 1, 0);


data_test  = dlmread("test_T.csv", ",", 1, 0);


data = [data_train; data_test];

x1 = data(:, 1);
x2 = data(:, 2);

label = data(:, 3);

figure;
hold on;

for c = 1:4
    idx = (label == c);
    scatter(x1(idx), x2(idx), 10, ".", "DisplayName", sprintf("C%d", c));
end

hold off;

legend("show");
axis([0 2 0 2]);
axis equal;
grid on;
title("Dataset ΣΔΤ (train + test, 4 κατηγορίες)");
xlabel("x_1");
ylabel("x_2");

%print("plot_T_all.png", "-dpng");

