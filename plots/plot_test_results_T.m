data = dlmread("test_results_T.csv", ",", 1, 0);

x1 = data(:,1);
x2 = data(:,2);
trueLabel = data(:,3);
predLabel = data(:,4);
correct   = data(:,5);

idx_ok  = (correct == 1);
idx_err = (correct == 0);

figure;
hold on;

plot(x1(idx_ok),  x2(idx_ok),  "+");

plot(x1(idx_err), x2(idx_err), "x");

xlabel("x1");
ylabel("x2");
title("Σύνολο ελέγχου: '+' σωστά, 'x' λάθος");
axis([0 2 0 2]);
grid on;
hold off;

