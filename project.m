data = readtable('student_depression_dataset.csv', 'VariableNamingRule', 'preserve');

vars = [ ...
    "Academic Pressure", "CGPA", "Sleep Duration", ...
    "Work/Study Hours", "Study Satisfaction", ...
    "Financial Stress", "Family History of Mental Illness", "Depression" ...
    ];
modelData = data(:, vars);

sleepStr = modelData.("Sleep Duration");
sleepNum = zeros(height(modelData), 1);
for i = 1:height(modelData)
    str = sleepStr{i};
    if contains(str, 'Less than')
        sleepNum(i) = 4.5;
    elseif contains(str, '5-6')
        sleepNum(i) = 5.5;
    elseif contains(str, '6-7')
        sleepNum(i) = 6.5;
    elseif contains(str, '7-8')
        sleepNum(i) = 7.5;
    elseif contains(str, '8-9')
        sleepNum(i) = 8.5;
    elseif contains(str, '9-10')
        sleepNum(i) = 9.5;
    elseif contains(str, 'More than')
        sleepNum(i) = 10.5;
    else
        sleepNum(i) = NaN;
    end
end
modelData.SleepDuration = sleepNum;

modelData.FinancialStress = double(categorical(modelData.("Financial Stress")));
modelData.FamilyHistory = double(categorical(modelData.("Family History of Mental Illness")));

modelData = rmmissing(modelData);

X = [ ...
    modelData.("Academic Pressure"), ...
    modelData.CGPA, ...
    modelData.SleepDuration, ...
    modelData.("Work/Study Hours"), ...
    modelData.("Study Satisfaction"), ...
    modelData.FinancialStress, ...
    modelData.FamilyHistory ...
    ];
y = modelData.Depression;

mdl = fitlm(X, y);
disp(mdl);

predictorNames = { ...
    "Academic Pressure", "CGPA", "Sleep Duration", ...
    "Work/Study Hours", "Study Satisfaction", ...
    "Financial Stress", "Family History" ...
    };
colors = lines(length(predictorNames));  % Unique colors

figure;
hold on;

for i = 1:length(predictorNames)
    base = mean(X);
    
    x_range = linspace(min(X(:,i)), max(X(:,i)), 100)';
    
    X_new = repmat(base, 100, 1);
    X_new(:,i) = x_range;
    
    y_pred = predict(mdl, X_new);
    plot(x_range, y_pred, 'LineWidth', 2, 'Color', colors(i,:), ...
        'DisplayName', predictorNames{i});
end

xlabel('Predictor Value');
ylabel('Predicted Depression Level');
title('Effect of Each Predictor on Depression (Regression Lines)');
legend('Location', 'best');
grid on;
