
	Hey there!

So you want to run some classifier experiments, perhaps comparing performance
across different classifiers and data formats? This README provides a detailed 
account of how to use the contents of the src folder to format time series data
and run classifiers on that data. This document should be updated as bugs and 
logistical issues are overcome. 

*------------------------------------------------------------------------------*
	
	Formatting data

This folder contains a slew of data formatting python scripts. They are focused
on working with numerical time series data. The data should exist in two files 
that contain the per-time-step (csv) and metadata (csv), and a data dict (json)
that describes the purposes of each data column. 

	- The data dict provides information on what each column in the data
	contains. This means its name, description, role, data-type, and 
	constraints. Generally, name and role are most important. The "role" value
	in particular is used to differentiate the "id" columns that link a certain
	time series together, from the "output" columns that contain your labels, 
	from the "feature" columns that contain input values. There are also 
	"sequence" columns that indicates what time step a row pertains to. Some of 
	the formatting tools will change, add, or remove columns, and thus will 
	produce an  updated data dict each time they run. 
	
	- The per-time-step data is the raw readings for each unit of time in the
	time series data. This file must feature "id", "feature", and "sequence"
	columns. The "id" and "sequence" columns must be populated in every row. 
	The "feature" columns do not need to have values. If the "outcome" of an 
	episode is static it may be included in the metadata file instead, but if
	it changes over time, it must be inlcuded and populated. 
	
	- The metadata file is a little more dynamic. Like the per-time-step data
	file, it must include all "id" columns, fully populated. And as mentioned,
	it may contain populated "outcome" columns. It is otherwise used for static
	data that does not change over time. All "id" combinations in the per-time-
	step data must exist in the metadata file, but the metadata file may
	contain metadata for unique series that do not exist in the per-time-step
	file. 

The scripts themselves are fairly straightforward if the data files are all in 
alignment. There are two categories of data pipeline, one for time-series
compatible classifiers, such as LSTMs, and one for classifiers that produce a 
prediction for a single feature vector, i.e. linear classifiers and vanilla 
neural networks. The pipeline is designed so that one can easily use both and 
compare the results of different classifiers on the same data. The following
five scripts can most easily be used in order, but should all function as 
standalone scripts. They each have specific inputs that can be viewed in their
source code, I recommend using ctrl-f to search for "parser.add_arguments" 
which will highlight each input and its function in detail. 

	align_to_grid.py - This script takes time series that were recorded at 
	variable time steps and aligns them to a single, uniform time step. The
	script takes in per-time-step data, a data dict, a step size argument, 
	and an output file location. While the other inputs are straightforward,
	the step size argument requires some explanation. Step size must be an 
	integer larger than 0. 
		- When given "1", the script will find the smallest time step 
		differential and use that as its uniform time step. This means that 
		larger time steps will be broken up into smaller ones, some of which
		are have empty values. An original time step may fall in between two 
		uniform time steps; in this case, that data will fall into the 
		latter of the two steps it falls between. 
		- When given a time step larger than 1, say "10", the uniform time 
		step will be 10 times the smallest time step. If such a time step
		encompasses multiple original time steps, then those values will be
		averaged and assigned to the latter of their encasing, uniform time 
		steps. Again, some values may be empty. 

	fill_missing_values.py - This script looks to solve the problem created 
	by the align_to_grid.script, in which some of the uniform time steps 
	never corresponded to any recorded data. Thus, they must be filled. This
	script offers several strategies for doing so. To use multiple strategies
	in series, one must set the --multiple_strategies flag to True. 
		- "pop_mean" simply takes the average column value for the entire
		dataset and inserts it into missing values in that column. The only
		way this does not fill every row of the column is if no row has a 
		value from which to derive a baseline average.
		- "carry_forward" takes the most recent value by time step, and 
		inserts it into every proceeding empty step. It essentially fills in
		the missing data by assuming a value goes unchanged until it is 
		recorded changing. This can fail to fill any rows if that time-series
		has no data point for that column. It can often fail to fill the first
		few rows if the very first time step is not completely filled out. 
		- "nulls" replaces every empty value with a 0. 
		- "similar_subject_mean", "GRU_simple", and "GRU_complex" are planned
		strategies, but are currently not implemented in a useful form. 

	normalize_features.py - This script simply takes all the values in each 
	column of the per-time-step data and normalizes them against the population
	averages. They are normalized on a robust scale, based on each columns'
	inter-quartile range. 

	feature_transformation.py - This script has two primary functionalities,
	adding new columns and collapsing time series.  
		- The --add_feature flag invokes adding a column to an existing 
		dataset, such that the values of the new column are a function of the
		values of a specified, pre-existing column. Currently, only simple 
		mathematical functions are implemented. Eventually, user defined 
		functions should be implemented.
		- The --collapse flag indicates collapsing the individual time-series
		in a dataset into a single feature vector. This use allows one to use 
		classifiers that require linear input. In general, this will take the 
		form of taking each of the original feature columns and creating a 
		series of summary statistics. There are a range of options for
		how this can be done.
			- The --collapse_features flag specifies which mathematical summary
			statistics are desired. Each function listed in the script input 
			is run on every feature column and added to a new data file.
			- The --collapse_range_features flag specifies which mathematical
			summary statistics are desired on specific ranges. Each function
			listed here is applied to every column, and every range of those
			columns. 
			- The --range_pairs flag specifies those ranges. This allows one to
			take summaries of the whole time-series (0, 100), or the middle 50%
			of data points (25, 75), and so on. Each range is applied to every
			column. 
			- The --max_time_step flag provides an upper bound for time steps
			summarized. So if your time step is equivalent to 1 hour, and the
			max_time_step is 48, your data will be collapsed over the first 
			2 days. This means that the ranges specified by the range_pairs
			flag will be ranges of those 48 hours (as in, (25, 75) indicates
			hours 12 to 36).
		This can quickly lead to an explosion in the number of columns, the 
		total of which is: 

			[(num of original columns) * (num of collapse features)]
	  	  + [(num of original columns) * (num of ranges) 
	  	  	                           * (num of range features)]

	  	Furthermore, pay special attention to the --data_dict_output flag, and
	  	make sure to use the updated data dict with the newly created data 
	  	file. This will keep track of all the new columns, whose names are 
	  	combinations of the original column name, their range, and operation. 
	  	WARNING: Some operations, such as slope, require at least two
	  	datapoints to output a value other than 0. 

	split_dataset.py - This script takes a data file, collapsed or time-series, 
	and splits it into two data files, one test data and one training data. One
	can specify the percentage split, random seed, and the columns to group by.
		-The columns to group by (--group_cols) helps prevent different 
		versions of related time-series from appearing in test and train. For 
		example, say you have data from the same subject, two years apart. The
		"subject_id" and "year" columns are your id columns, so they are 
		considered unique. Including one in your training data and one in your 
		test data can compromise the validity of your results. Instead, you can
		specify "--group_by subject_id" to ensure that both entries for that 
		subject end up in the same dataset. 


*------------------------------------------------------------------------------*

	Testing classifiers

We have built three classifier experiments that serve as good examples for how 
to use our data pipeline setup to measure classifier performance. First, let us
go over a linear classifier in order to learn how to read data in, grid search 
over training data, measure performance on test data, and compile test results 
into a summarizing report. Then, we will dive into how we integrate PyTorch's 
neural networks into the sklearn GridSearchCV for more advanced classifiers.

	Linear classifier - Running experiments on a collapsed time-series dataset
	is fairly straightforward because we can use sklearn for pretty much 
	everything we want to do. 

		1. Input processing: We want to import our training and test data. If
		you use the pipeline described in the first half of this README, you 
		will need to include the test and train csv files as well as the 
		metadata file and data dict. Depending on whether outcomes are stored 
		in the test and train files or the metadata files, you will need to 
		write a small function that extracts the labels for each dataset. 
		Eventually you will need to end up with X_train and y_train, and X_test 
		and y_test. You may also want to create command line argument with 
		argparse that allows you to input the hyperparameters, rather than 
		hardcoding them. 

		2. Training and testing: From here you can follow the examples in the 
		sklearn documentation closely. Create an instance of your classifier 
		type and pass it to GridSearchCV() to create your classifier. Then run
		the fit() method on your training data. After the best hyperparameters 
		are discovered, you will be left with a classifier object trained on
		all the training data (not just the fold that validated highest). 
		From here you will want to evaluate its performance on the test data. 
		Use the predict() method for one-hot output, and the predict_proba()
		method for probabilistic output (note: one-hot can be derived from
		probabilistic using a threshold of your choosing). If AUROC or log 
		loss are some of the performance measures you wish to report, you will 
		need the probabilistic output on hand. 

		3. Measuring performance: Most functions for producing performance 
		measures can be found in the sklearn.metrics package. From there you 
		can compute log loss, balanced accuracy, as well as create confusion 
		matrices and ROC curves, among other measures. Pay extra attention to 
		the ordering of your output labels, as it is important not to mix up 
		positive and negative outcomes when comparing to y_test. 

		4. We used yattag to create a small html file that held numerical 
		performance measures, descriptions of the classifier's parameters, and
		references to matplotlib images, all saved in the same --report 
		directory. 

	RNN - Experimenting on a RNN requires a bit more work, especially if you 
	want to use sklearn gridsearch. In the src/rnn folder, you will find 
	implementations of a working LSTM that plugs into sklearn nicely. This 
	features dataset_loader.py for packaging each time-series up nicely, and 
	RNNBinaryClassifier and RNNBinaryClassifierModule, which leverage the 
	skorch package to bridge the gap between PyTorch and sklearn. It is not 
	necessary to copy these classes exactly, but they are a helpful place to
	get started. 

		1. dataset_loader.py exports a useful class called 
		TinySequentialDataCSVLoader. This class takes in per-time-step data,
		its metadata, a list containing "id" column names, the name of the 
		output column, and "y_label_type" which indicates whether the output
		is static or changes over time. A good example can be found in 
		src/rnn/main_mimic.py. Once initiated, this class instance will hold
		all of your data, which can be accessed a single, specified time-series
		at a time using the get_single_sequence_data() class methods, or all at 
		once using the get_batch_data() class method. For integration with 
		GridSearchCV, use get_batch_data() to create your (X, y) train and 
		test data. 

		2. RNNBinaryClassifierModule contains the implementation of an LSTM
		using PyTorch. There isn't anything too special going on here: many 
		examples of similar neural networks can be found online and adapted to 
		your specifications. The only unique change that needs to be made is 
		to add a score() class method that takes in a list of time-series and
		the corresponding labels, and returns the percentage of correctly 
		predicted labels as a float. This is the method that GridSearchCV 
		uses to grade each hyperparameter and cross fold combination. 

		3. RNNBinaryClassifier is a wrapper that uses skorch to build an 
		interface compatible with sklearn on top of the PyTorch implementation.
		It may appear a little inscrutible at first, but as far as I can tell,
		the implementation of __init__() found in RNNBinaryClassifier.py is the
		standard way to wrap a neural network using skorch, and should be 
		followed closely. You are essentially telling the skorch.NeuralNet 
		module where to look for certain arguments and methods. To access our
		score method in particular (one that skorch does not link up for us), 
		we need to invoke self.module_.score(). All of the methods in your 
		PyTorch implementation can be accessed through the self.module_ 
		pattern. 

		When initializing your classifier instance so that you can pass it to
		GridSearchCV, you will need to change some argument names to:
		module__<argument name> so that skorch recognizes them as headed to the
		PyTorch implementation. 

	Beyond that, the implementation is the same as it is in the linear 
	classifiers. Hooking skorch and the PyTorch class up correctly can be 
	tricky, but once finished, everything should work like a charm. 


*------------------------------------------------------------------------------*

	Setting up environments and using the Tufts cluster

Fortunately, we can run big experiments by using the Tufts high performance
cluster. You will need to talk to Prof. Hughes to get access, but I have 
otherwise detailed my process for getting everything to run here, so that 
hopefully you can save some time. 

	Cluster: Once you have access to the HPC, I recommend following the new 
	user guide found here: https://tufts.app.box.com/v/HPC-New-User and 
	https://www.cs.tufts.edu/comp/150BDL/2018f/tufts_hpc_setup.html 

	Environments: We have been using the conda environment py3.6_torch1.0_pyro0.3 which provides everything we need to use PyTorch,
	skorch, and sklearn. I recommend against setting up your own conda 
	environment as there are some hurdles to overcome when installing 
	packages from scratch on the HPC, and a private conda may mess with your
	.bashrc file. 

	Pipeline scripts: I was best able to organize our data pipeline, from raw
	to data formatting to measuring classifier performance, by creating bash
	scripts. A fairly straightforward example a script can be found in 
	src/rnn/full_pipeline.sh. The most important part of each script is to make 
	sure the starting files are the right ones (I got this wrong a few times 
	when I would go between my local, small test sets, and the large test sets 
	on the cluster), and all the directories and files do in fact exist. I 
	recommend automating this process. I also recommend configuring your 
	pipeline to have a shortcut that allows you to skip data formatting and run
	just the classifier. Often, data will format correctly, but the classifier 
	will fail with a bug, and it saves a lot of time to not reformat the data.

	Submitting workloads: Generally, although it is possible to open a session
	with the srun command, I recommend submitting a small script to sbatch 
	instead. This will save you the time of initializing a session and 
	activating your environment by hand everytime. Furthermore, it allows 
	others to quickly replicate what you have done. 
	
	One quirk of submitting a workload to the HPC is that you have to provide 
	an estimate of how long you think it will take to finish. The upper limit
	is 72 hours (-t 0-72:00:00), and if you have never run your classifier on 
	a certain dataset before, I recommend using the upper limit and adjusting 
	your estimation down from there. Depending on your setup, scripts can take
	anywhere from a few hours to almost 3 days to run. 

	For example, our LSTM pipeline takes about 12 hours to complete when run on 
	the mimic3 dataset. This is with 100 cores requested and a large batch size 
	(256). When it was initially run with a very small batch size (3) and only 
	10 cpus, it timed out (do not use a batch size of 3, this is way too 
	small). The random forest pipeline was faster than 12 hours, and the 
	logistic pipeline was even faster than that. 

	A note on cpu nodes, I believe you are allowed to request up to 412 nodes,
	but this is far more than is realistically needed. 100 seems to be a good 
	amount for the work we are doing. 


*------------------------------------------------------------------------------*

Update log: 
Oliver Newland 8/21/19


