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

net = newff(TrainData, TrainDataTargets, [30],{},'traingd');

% Set Parameters values for traing
net.divideParam.trainRatio=1;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;
net.trainParam.epochs=1;
net.trainParam.lr = 0.2;

% Initialization
lambda = 0.001;
d = 0.005;
nonZero = zeros(10000, 1);
performance = [];

for k=1:1000
    oldWeights = getwb(net);
    [net,tr] = train(net,TrainData,TrainDataTargets);
    newWeights = getwb(net)
    newWeights = newWeights - lambda * oldWeights;

		% Set equal with 0 the weights which are smaller than d
		for i = 1:size(newWeights)
			if(abs(newWeights(i)) < d)
				newWeights(i) = 0;
			end
		end

		% Count the non zero weights every iteration
		for i = 1:size(newWeights)
        if (abs(newWeights(i)) > 0)
            nonZero(k) = nonZero(k) + 1;
        end
    end

		% Load the new weights to network
    net = setwb(net,newWeights);
    performance = [performance tr.perf(2)];
end

% Plot nonZero and performance
figure(3);
plot(nonZero, 'LineWidth',2);
figure(4);
plot(performance, 'LineWidth',2);
% Calculate performance
TestDataOutput = sim(net, TestData);
[accuracy, precision, recall] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);


fprintf('accuracy: %f\n', accuracy);
fprintf('precision: %f\n', precision);
fprintf('recall: %f\n', recall);
%% =============================================================

%% =============================================================
