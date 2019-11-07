# Contents

* main.py
* * Call with no args to train an RNN on a simple dataset via SGD, then make per-sequence probabilistic predictions

* dataset_loader.py
* * Provides a useful class to load x,y data from .csv files in our tidy sequential dataset format
* * Call with no args to test

* RNNBinaryClassifierModule.py
* * Defines the core RNN functionality. Thin wrapper around pytorch's RNN.

* RNNBinaryClassifier.py
* * Wraps the core RNN functionality into an object that behaves like sklearn classifiers
* * * Call .predict(X) with numpy array X to make predictions
* * * Call .fit(X, y) with appropriate numpy arrays X, y to fit the RNN to training data (X,y)
* * * Call .partial_fit(X, y) with appropriate numpy arrays X, y to do a single gradient step update
* * Thin wrapper around skorch's Net object
* * For more, see: https://github.com/skorch-dev/skorch/blob/master/skorch/net.py#L40


# Resources

Checkout the examples on skorch project github:

* https://github.com/skorch-dev/skorch/tree/master/examples/word_language_model
* https://github.com/skorch-dev/skorch/blob/master/examples/rnn_classifer/RNN_sentiment_classification.ipynb

Note that these are for models of natural language, so the data representation/ first layer is often very different than our purposes

