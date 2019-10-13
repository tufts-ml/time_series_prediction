%%
% The code is written in Matlab and tested on a Ubuntu 14.04 machine with Matlab 2014b.
% To repeat the experiments: open the matlab script "evall.m", change the variables "datapath",
% "splitpath" and "resultpath" in agreement with your local path. 
% To repeat the same numbers of the paper, check if the original training/test splits are in the ù
% folder "./data/split/".


datapath = '/home/fender/Dropbox/UniMiB-SHAR/data/';
splitpath = 'split';
resultpath = 'results';

%% knn evaluation

eval_multiclass(datapath,splitpath,resultpath,'two_classes','kfold','knn');
eval_multiclass(datapath,splitpath,resultpath,'acc','kfold','knn');
eval_multiclass(datapath,splitpath,resultpath,'adl','kfold','knn');
eval_multiclass(datapath,splitpath,resultpath,'fall','kfold','knn');


eval_multiclass(datapath,splitpath,resultpath,'two_classes','subjective','knn');
eval_multiclass(datapath,splitpath,resultpath,'acc','subjective','knn');
eval_multiclass(datapath,splitpath,resultpath,'adl','subjective','knn');
eval_multiclass(datapath,splitpath,resultpath,'fall','subjective','knn');

%% svm evaluation

eval_multiclass(datapath,splitpath,resultpath,'two_classes','kfold','svm');
eval_multiclass(datapath,splitpath,resultpath,'acc','kfold','svm');
eval_multiclass(datapath,splitpath,resultpath,'adl','kfold','svm');
eval_multiclass(datapath,splitpath,resultpath,'fall','kfold','svm');


eval_multiclass(datapath,splitpath,resultpath,'two_classes','subjective','svm');
eval_multiclass(datapath,splitpath,resultpath,'acc','subjective','svm');
eval_multiclass(datapath,splitpath,resultpath,'adl','subjective','svm');
eval_multiclass(datapath,splitpath,resultpath,'fall','subjective','svm');



