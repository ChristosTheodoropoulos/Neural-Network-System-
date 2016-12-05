%% Νευρωνικά Δίκτυα και Ευφυή Υπολογιστικά Συστήματα
%% Μελέτη των πολυεπίπεδων Perceptrons και εφαρμογή σε προβλήματα ταξινόμησης εικόνας

%% Αρχικοποίηση
clear ; close all; clc

%% ========== Μέρος 1: Προεπεξεργασία των δεδομένων ============

%% ========================= ΒΗΜΑ 1 ============================
% Load Data
fprintf('Loading and Visualizing Data ...\n')

load('dataSet.mat');

% Διαστάσεις του κάθε πίνακα
NumberOfRowsTrainDataTargets = size(TrainDataTargets, 1);
NumberOfColumnsTrainDataTargets = size(TrainDataTargets, 2);

NumberOfRowsTrainData = size(TrainData, 1);
NumberOfColumnsTrainData = size(TrainData, 2);

NumberOfRowsTestDataTargets = size(TestDataTargets, 1);
NumberOfColumnsTestDataTargets = size(TestDataTargets, 2);

NumberOfRowsTestData = size(TestData, 1);
NumberOfColumnsTestData = size(TestData, 2);


sizeOfCategory = sum(TrainDataTargets, 2);	% Μέγεθος της κάθε κατηγορίας
bar(sizeOfCategory);	% Εμφάνιση με μπάρες του μεγέθους κάθε κατηγορίας

tempSizeOfCategory = sizeOfCategory;	% Αντίγραφο του πίνακα sizeOfCategory
minCategory = min(sizeOfCategory);		% Μικρότερη σε μέγεθος κατηγορία

% Ο πίνακας index έχει άσσo αν θέλω να κρατήσω τη συγκεκριμένη στήλη
% και μηδέν αν δε θέλω να την κρατήσω. Εδώ τον αρχικοποιώ με άσσους.
index = ones(NumberOfColumnsTrainDataTargets, 1);

% Επανάληψη για να ξέρω ποιες στήλες του πίνακα TrainDataTargets θα
% κρατήσω έτσι ώστε να έχω ίδιο αριθμό δεδομένων από κάθε κατηγορία.
for i = 1:NumberOfColumnsTrainDataTargets
	for j = 1:NumberOfRowsTrainDataTargets
		if (TrainDataTargets(j,i) == 1 && tempSizeOfCategory(j) > minCategory)
			index(i) = 0;
			tempSizeOfCategory(j) = tempSizeOfCategory(j) - 1;
		end
	end
end

% Σβήσιμο των στηλών που δε θα χρησιμοποιήσω από το TrainDataTargets
% πίνακα και από το TrainData πίνακα.
counter = 0;
for i = 1:NumberOfColumnsTrainDataTargets
	if (index(i) == 0)
		TrainDataTargets(:, i - counter) = [];
		TrainData(:, i - counter) = [];
		counter = counter + 1;
	end
end


% Τσεκάρισμα αν έκοψα τις σωστές στήλες.
checksizeOfCategory = sum(TrainDataTargets, 2);
figure(2);
bar(checksizeOfCategory);
%% =============================================================
%% ========================= ΒΗΜΑ 2 ============================
% Προεπεξεργασία των δεδομένων των πινάκων TrainData και TestData
% με χρήση των συναρτήσεων removeconstantrows, mapstd, processpca
% κατά σειρά.
[TrainData, PS1] = removeconstantrows(TrainData);
TestData = removeconstantrows('apply', TestData, PS1);

[TrainData, PS2] = mapstd(TrainData);
TestData = mapstd('apply', TestData, PS2);

[TrainData, PS3] = processpca(TrainData, 0.008);
TestData = processpca('apply', TestData, PS3);
%% =============================================================

%% =============================================================
%% = Μέρος 2: Μελέτη της Αρχιτεκτονικής του Νευρωνικού Δικτύου =
%% ========================= ΒΗΜΑ 3 ============================
% Κατασκευή Νευρωνικού Δικτύου MLP με χρήση της συνάρτησης newff.
% Χρησιμοποίησα εκπαίδευση με Early Stopping χρησιμοποιώντας το
% 80% των δεδομένων για εκπαίδευση και 20% για επαλήθευση.
% trainRatio = 0.8, valRatio = 0.2, testRatio = 0

net = newff(TrainData, TrainDataTargets, [10 20],{'tansig' 'tansig' 'tansig'},'traingd', 'learngdm');
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
net.trainParam.lr = 0.2;
net.trainParam.epochs = 10000;
net.trainParam.max_fail = 6;
%% =============================================================

%% ========================= ΒΗΜΑ 5 ============================
% Εκπαίδευση του Νευρωνικού Δικτύου

[net, tr] = train(net, TrainData, TrainDataTargets);
TestDataOutput = sim(net, TestData);
[accuracy, precision, recall] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);

fprintf('accuracy: %f\n', accuracy);
fprintf('precision: %f\n', precision);
fprintf('recall: %f\n', recall);
%% =============================================================

%% =============================================================
