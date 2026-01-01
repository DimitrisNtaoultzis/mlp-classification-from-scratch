data = dlmread("test_results_T.csv", ",", 1, 0);
x1 = data(:,1);
x2 = data(:,2);
trueLabel = data(:,3);
correct   = data(:,5);

idx1 = (trueLabel == 1);
idx2 = (trueLabel == 2);
idx3 = (trueLabel == 3);
idx4 = (trueLabel == 4);
idx_err = (correct == 0);

figure; hold on;
plot(x1(idx1), x2(idx1), "r+");
plot(x1(idx2), x2(idx2), "g+");
plot(x1(idx3), x2(idx3), "b+");
plot(x1(idx4), x2(idx4), "m+");

plot(x1(idx_err), x2(idx_err), "kx", "markersize", 8);

title("Κατηγορίες C1..C4 και λάθη (μαύρο 'x')");
axis([0 2 0 2]); grid on; hold off;

