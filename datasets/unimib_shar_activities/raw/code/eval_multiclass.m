% /*************************************************************************************
%
%    Project Name:  UniMiB SHAR: a new dataset for human activity recognition using acceleration data from smartphones
%    File Name:     eval_multiclass.m
%    Authors:       D. Micucci and M. Mobilio and P. Napoletano (paolo.napoletano@disco.unimib.it)
%    History:       October 2016 created
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function implements the evalaution of the UniMiB SHAR dataset using KNN and SVM.
%
% USAGE
%  final_results = eval_multiclass(path,split_path,results_path,data_type,type_eval,classifier_type)
%
% INPUT
%  path = accelerometer datapath
%  split_path = training/test splits datapath
%  results_path = path of the folder containing the results
%  data_type = type of accelerometer splits: 'two_classes','acc','adl',
%  'fall'
%  type_eval = 'kfold' or 'subjective'
%  classifier_type = KNN or SVM
%
% OUTPUT
%  final_results    - cell [n x 3] containing the SA, MAA, ConfusionMatrix 
%
%


function [final_results] = eval_multiclass(path,split_path,results_path,data_type,type_eval,classifier_type)

final_results = [];
switch type_eval
    case 'kfold'
        final_results = kfold_eval(path,split_path,results_path,5,data_type,classifier_type)
    case 'subjective'
        final_results = subjective_eval(path,split_path,results_path,data_type,classifier_type)
    otherwise
        disp('error typing type of evaluation')
end

end

function final_results = kfold_eval(path,split_path,results_path,num_split,data_type,classifier_type)


class_names = [data_type '_names'];

data_names_fn = fullfile(path,[class_names '.mat']);

load(data_names_fn,class_names)

data_name = [data_type '_data'];
data_labels = [data_type '_labels'];

data_fn = fullfile(path,[data_name '.mat']);
labels_fn = fullfile(path,[data_labels '.mat']);

load(data_fn,data_name)
load(labels_fn,data_labels)

loaded_class_names = eval(class_names);
loaded_data = eval(data_name);
loaded_labels = eval(data_labels);

num_classes = size(unique(loaded_labels(:,1)),1);
num_subjects = size(unique(loaded_labels(:,2)),1);
%start = 1:3:size(loaded_data,1);
%stop = 3:3:size(loaded_data,1);
%new_data = [];

%for ii=1:size(start,2)
%    ii
%    new_data = [new_data; reshape(loaded_data(start(ii):stop(ii),:)',[],size(loaded_data,2)*3)];
%end
%loaded_labels = loaded_labels(start,:);
%loaded_data = new_data;

splitted_data_fn = fullfile(path,split_path,[data_type '_splitted_data' num2str(num_split) '_folds.mat']);
train_idxs_fn = fullfile(path,split_path,[data_type '_train_idxs' num2str(num_split) '_folds.mat']);
test_idxs_fn = fullfile(path,split_path,[data_type '_test_idxs' num2str(num_split) '_folds.mat']);

splitted_data = {};
train_idxs = {};
test_idxs = {};

if((~exist(splitted_data_fn))||(~exist(train_idxs_fn))||(~exist(test_idxs_fn)))
    
    for l=1:num_classes
        
        class_labels = loaded_labels(:,1) ==l;
        class_adl_data = loaded_data(class_labels,:);
        current_prob_class = sum(class_labels);
        step = floor(current_prob_class/num_split);
        current_prob_class = step*num_split;
        start = 1:step:current_prob_class;
        
        splitted_data{l,1}=class_adl_data(1:current_prob_class,:);
        
        adl_index = randperm(current_prob_class);
        
        class_test_idx = [];
        class_train_idx = [];
        
        for ss = 1:size(start,2)
            current_test_idx = adl_index(start(ss):start(ss)+step-1);
            
            class_test_idx = [class_test_idx; current_test_idx];
            class_train_idx = [class_train_idx; setdiff(adl_index,current_test_idx)];
            
        end
        train_idxs{l,1} = class_train_idx;
        test_idxs{l,1} = class_test_idx;
        
    end
    
    save(splitted_data_fn,'splitted_data')
    save(train_idxs_fn,'train_idxs')
    save(test_idxs_fn,'test_idxs')
    
    
else
    
    load(splitted_data_fn,'splitted_data')
    load(train_idxs_fn,'train_idxs')
    load(test_idxs_fn,'test_idxs')
    
    
end


final_results = {};

for eva=1:num_split
    % eva
    current_train_data = [];
    current_test_data = [];
    current_train_labes = [];
    current_test_labes = [];
    
    for l=1:num_classes
        
        current_class_data = splitted_data{l,1};
        class_train_idxs = train_idxs{l,1};
        current_train_data  = [current_train_data; current_class_data(class_train_idxs(eva,:),:)];
        
        class_test_idxs = test_idxs{l,1};
        current_test_data  = [current_test_data; current_class_data(class_test_idxs(eva,:),:)];
        current_train_labes = [current_train_labes; ones(size(class_train_idxs,2),1)*l];
        current_test_labes = [current_test_labes; ones(size(class_test_idxs,2),1)*l];
        
    end
    
    results_fn = fullfile(path,results_path,['results_' data_type '_' classifier_type '_' num2str(num_split) '_folds.mat']);
    
    switch classifier_type
        case 'svm'
            SVMModels = cell(num_classes,1);
            Scores = zeros(size(current_test_data,1),num_classes);
            fprintf('training split n.%d with svm\n',eva)
            
            for j = 1:num_classes;
                indx = current_train_labes==j;
                SVMModels{j} = fitcsvm(current_train_data,indx,'Standardize',true,'KernelFunction','RBF',...
                    'KernelScale','auto');
                %SVMModels{j} = crossval(SVMModels{j});
                [~,score] = predict(SVMModels{j},current_test_data);
                Scores(:,j) = score(:,2);
                
            end
            fprintf('testing split n.%d with svm\n',eva)
            
            [~,predictions] = max(Scores,[],2);
            results = predictions ==current_test_labes;
            class_accuracy = [];
            for l=1:num_classes
                true_positives = current_test_labes == l;
                class_accuracy = [class_accuracy; sum(predictions(true_positives) == l)/size(test_idxs{l,1},2)];
                
            end
            
            final_results{eva,1} = (sum(results))/size(current_test_labes,1);
            final_results{eva,2} = mean(class_accuracy);
            final_results{eva,3} = confusionmat(current_test_labes,predictions);
            
            
        case 'knn'
            fprintf('training split n.%d with knn\n',eva)
            KNNModel = fitcknn(current_train_data,current_train_labes,'NumNeighbors',1);
            fprintf('testing split n.%d with knn\n',eva)
            
            predictions = predict(KNNModel,current_test_data);
            results = predictions ==current_test_labes;
            class_accuracy = [];
            for l=1:num_classes
                true_positives = current_test_labes == l;
                class_accuracy = [class_accuracy; sum(predictions(true_positives) == l)/size(test_idxs{l,1},2)];
                
            end
            
            final_results{eva,1} = (sum(results))/size(current_test_labes,1);
            final_results{eva,2} = mean(class_accuracy);
            final_results{eva,3} = confusionmat(current_test_labes,predictions);
        otherwise
            disp('please choose a valid classifier')
            
            
    end
    
    
end

save(results_fn,'final_results')

disp('evaluation_completed')

end


function final_results = subjective_eval(path,split_path,results_path,data_type,classifier_type)


class_names = [data_type '_names'];

data_names_fn = fullfile(path,[class_names '.mat']);

load(data_names_fn,class_names)

data_name = [data_type '_data'];
data_labels = [data_type '_labels'];

data_fn = fullfile(path,[data_name '.mat']);
labels_fn = fullfile(path,[data_labels '.mat']);

load(data_fn,data_name)
load(labels_fn,data_labels)

loaded_class_names = eval(class_names);
loaded_data = eval(data_name);
loaded_labels = eval(data_labels);

num_classes = size(unique(loaded_labels(:,1)),1);
num_subjects = size(unique(loaded_labels(:,2)),1);
%start = 1:3:size(loaded_data,1);
%stop = 3:3:size(loaded_data,1);
%new_data = [];

%for ii=1:size(start,2)
%    new_data = [new_data; reshape(loaded_data(start(ii):stop(ii),:)',[],size(loaded_data,2)*3)];
%end
%loaded_labels = loaded_labels(start,:);
%loaded_data = new_data;

splitted_data_fn = fullfile(path,split_path,[data_type '_splitted_data' 'subjective' '_folds.mat']);
train_idxs_fn = fullfile(path,split_path,[data_type '_train_idxs' 'subjective' '_folds.mat']);
test_idxs_fn = fullfile(path,split_path,[data_type '_test_idxs' 'subjective' '_folds.mat']);

splitted_data = {};
train_idxs = {};
test_idxs = {};

if((~exist(splitted_data_fn))||(~exist(train_idxs_fn))||(~exist(test_idxs_fn)))
    indexes_vector = 1:1:size(loaded_labels,1);
    for l=1:num_subjects
        
        subject_labels = loaded_labels(:,2) ==l;
        no_subject_labels = ~subject_labels;
        
        train_idxs{l,1} = indexes_vector(no_subject_labels)';
        test_idxs{l,1} = indexes_vector(subject_labels)';
        
    end
    
    save(train_idxs_fn,'train_idxs')
    save(test_idxs_fn,'test_idxs')
    
    
else
    
    load(train_idxs_fn,'train_idxs')
    load(test_idxs_fn,'test_idxs')
    
    
end


final_results = {};

for l=1:num_subjects
    % eva
    current_train_data = [];
    current_test_data = [];
    current_train_labes = [];
    current_test_labes = [];
    
    %  for l=1:num_classes
    
    
    class_train_idxs = train_idxs{l,1};
    class_test_idxs = test_idxs{l,1};
    
    
    current_train_data = loaded_data(class_train_idxs,:);
    current_test_data = loaded_data(class_test_idxs,:);
    current_train_labes = loaded_labels(class_train_idxs,1);
    current_test_labes = loaded_labels(class_test_idxs,1);
    
    %  end
    
    %disp('training and testing')
    results_fn = fullfile(path,results_path,['results_' data_type '_' classifier_type '_subjective.mat']);
    
    switch classifier_type
        case 'svm'
            SVMModels = cell(num_classes,1);
            Scores = zeros(size(current_test_data,1),num_classes);
            fprintf('training split n.%d with svm\n',l)
            
            for j = 1:num_classes;
                indx = current_train_labes==j; % Create binary classes for each classifier
                SVMModels{j} = fitcsvm(current_train_data,indx,'Standardize',true,'KernelFunction','RBF',...
                    'KernelScale','auto');
                %SVMModels{j} = crossval(SVMModels{j});
                [~,score] = predict(SVMModels{j},current_test_data);
                Scores(:,j) = score(:,2); % Second column contains positive-class scores
                
            end
            fprintf('testing split n.%d with svm\n',l)
            
            [~,predictions] = max(Scores,[],2);
            results = predictions ==current_test_labes;
            class_accuracy = [];
            for j=1:num_classes
                true_positives = current_test_labes == j;
                class_current_test_labes = sum(true_positives);
                class_accuracy = [class_accuracy; sum(predictions(true_positives) == j)/class_current_test_labes];
                
            end
            class_accuracy(isnan(class_accuracy)) = 0;
            final_results{l,1} = (sum(results))/size(current_test_labes,1);
            final_results{l,2} = mean(class_accuracy);
            final_results{l,3} = confusionmat(current_test_labes,predictions);
            
        case 'knn'
            fprintf('training split n.%d with knn\n',l)
            KNNModel = fitcknn(current_train_data,current_train_labes,'NumNeighbors',1);
            fprintf('testing split n.%d with knn\n',l)
            
            predictions = predict(KNNModel,current_test_data);
            results = predictions ==current_test_labes;
            class_accuracy = [];
            for j=1:num_classes
                true_positives = current_test_labes == j;
                class_current_test_labes = sum(true_positives);
                class_accuracy = [class_accuracy; sum(predictions(true_positives) == j)/class_current_test_labes];
                
            end
            %             if(sum(isnan(class_accuracy))>0)
            %                 disp('please choose a valid classifier')
            %
            %             end
            class_accuracy(isnan(class_accuracy)) = 0;
            
            final_results{l,1} = (sum(results))/size(current_test_labes,1);
            final_results{l,2} = mean(class_accuracy);
            final_results{l,3} = confusionmat(current_test_labes,predictions);
            
        otherwise
            disp('please choose a valid classifier')
            
            
    end
    
    
end

save(results_fn,'final_results')

disp('evaluation_completed')

end

