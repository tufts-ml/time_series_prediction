# PC-VAE
Prediction-Constrained Variational Autoencoders

## Running experiments

Running an experiment is as simple as calling the `run_trials` function from `pcvae.experiments`. This function takes a dataset class, a model class and any of the corresponding options as listed by `list_options`:


```python
results = run_trials(dataset=fashion_mnist, model=PC, 
          num_labeled=100, epochs=5,
          seed=984, balance=True, batch_size=200)
```

## Models and datasets

The library includes a number of different models and datasets. We can check what is available using the `list_models` and `list_datsets` functions from `pcvae.util`:


```python
list_models()
list_datasets()
```

    Available models:
    	PC
    	Autoencoder
    	M1
    	ConsistantPC
    	M2
    	ADGM
    	SDGM
    	DNN
    	VAT
    
    Available datasets:
    	fashion_mnist
    	kingma_mnist
    	cifar
    	svhn
    	shar
    	caltech_birds
    
    


We can see the options available for a given model and dataset using `pcvae.util.list_options`:


```python
list_options(dataset=fashion_mnist, model=PC)
```



## Visualization

We can use the result object to make various plots with functions from `pcvae.visualizations`. For example: we can use the `plot_history` function to plot a trace of the training loss or accuracy over the training epochs:


```python
plot_history(results)
# Alt.: plot_history(results, metric='acc')
```


![png](tutorial_files/Tutorial_11_0.png)


Use `plot_confusion` to display a confusion matrix:


```python
plot_confusion(results, split='test')
```


![png](tutorial_files/Tutorial_13_0.png)


Use `plot_encodings` to display the test observations projected into the latent space. If the latent space is more than 2 dimensions, it will plot the first 2 principal components:


```python
plot_encodings(results, split='test')
```


![png](tutorial_files/Tutorial_15_0.png)


Use `plot_reconstructions` to display the test observations reconstructed through the model:


```python
plot_reconstructions(results, split='test')
```


![png](tutorial_files/Tutorial_17_0.png)


## Optimizing hyperparameters

We can also use the `run_trials` function to optimize the hyperparameters for the model. To do this we simply pass a search space for any number of hyperparameters. It will then run up to `n_trials` runs, using Bayesian optimization to select trial parameters:


```python
results = run_trials(dataset=kingma_mnist, model=PC,
          n_trials=10, num_labeled=100, epochs=10,
          
          LAMBDA     = uniform(1, 100),             # Uniform search space in specified range
          lr         = loguniform(1e-4, 1e-2),      # Logarithmic search space in specified range
          batch_size = integer(50, 200, step=50),   # Uniform search space with quantization
          balance    = categorical([True, False])   # Categorical search space
                              
          )
```

The results object will also carry information about all runs:


```python
results['results']['table']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_labeled</th>
      <th>epochs</th>
      <th>LAMBDA</th>
      <th>lr</th>
      <th>batch_size</th>
      <th>balance</th>
      <th>model</th>
      <th>train_acc</th>
      <th>valid_acc</th>
      <th>test_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>44.630241</td>
      <td>0.007120</td>
      <td>50</td>
      <td>False</td>
      <td>pc</td>
      <td>0.067688</td>
      <td>0.066921</td>
      <td>0.067649</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>18.009047</td>
      <td>0.000715</td>
      <td>150</td>
      <td>False</td>
      <td>pc</td>
      <td>0.056873</td>
      <td>0.056012</td>
      <td>0.056516</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>48.720489</td>
      <td>0.000752</td>
      <td>50</td>
      <td>True</td>
      <td>pc</td>
      <td>0.016901</td>
      <td>0.017144</td>
      <td>0.016927</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>14.498161</td>
      <td>0.001177</td>
      <td>50</td>
      <td>True</td>
      <td>pc</td>
      <td>0.018328</td>
      <td>0.018533</td>
      <td>0.018349</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>58.757713</td>
      <td>0.001487</td>
      <td>150</td>
      <td>False</td>
      <td>pc</td>
      <td>0.059250</td>
      <td>0.058376</td>
      <td>0.058973</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>3.387551</td>
      <td>0.000296</td>
      <td>150</td>
      <td>False</td>
      <td>pc</td>
      <td>0.049800</td>
      <td>0.049114</td>
      <td>0.049431</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>2.267173</td>
      <td>0.000339</td>
      <td>50</td>
      <td>True</td>
      <td>pc</td>
      <td>0.018185</td>
      <td>0.018271</td>
      <td>0.018033</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>48.757619</td>
      <td>0.000503</td>
      <td>100</td>
      <td>True</td>
      <td>pc</td>
      <td>0.018632</td>
      <td>0.018714</td>
      <td>0.018411</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>91.172600</td>
      <td>0.005541</td>
      <td>200</td>
      <td>False</td>
      <td>pc</td>
      <td>0.068684</td>
      <td>0.067318</td>
      <td>0.068448</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>10</td>
      <td>65.101542</td>
      <td>0.002557</td>
      <td>100</td>
      <td>True</td>
      <td>pc</td>
      <td>0.019598</td>
      <td>0.019857</td>
      <td>0.019625</td>
    </tr>
  </tbody>
</table>
</div>

## Running a hyperparameter grid

We can also explictly run all combinations of certain hyperparameters with `run_trials`. To do this, we specifiy each argument with the `grid` function. Trials will be run for all combinations of grid arguments. 

This can be combined with hyperpameter search spaces, in which case the hyperparameter optimizer will be run for each grid configuration.

```python
results = run_trials(dataset=kingma_mnist, 
          n_trials=10, epochs=10,
          model       = grid([PC, M2]),               # Run all 4 combinations of model and num_labeled            
          num_labeled = grid([100, 1000]),
          
          LAMBDA      = uniform(1, 100),             # Uniform search space in specified range
          lr          = loguniform(1e-4, 1e-2),      # Logarithmic search space in specified range
          batch_size  = integer(50, 200, step=50),   # Uniform search space with quantization
          balance     = categorical([True, False])   # Categorical search space
                              
          )
```

## Running remotely

We can also use the `run_remote` function to run an experiment on a remote machine. This function has the same interface as `run_trials`, but takes a remote_host and a unique experiment name as its first two arguments:

```python
from pcvae.util import remote_host
from pcvae.experiments import run_remote

host = remote_host(
    user              = 'username',                          # Username for remote host
    host              = 'ssh.myhost.uci.edu',                # URL for remote host
    datapath          = '/path/to/data/cache',               # Directory to store downloaded data on remote host
    TFPYTHONEXE       = '/users/me/miniconda3/bin/python',   # Python executable to use
    XHOST             = 'grid',                              # Flag to submit as grid job (use 'local' to run directly)
    XHOST_GPUS        = '1',                                 # GPUs to request for grid job
    XHOST_MEM_MB      = '4000',                              # Memory to request for grid job
    PCVAEROOT         = '/home/user/Research/Code/PC-VAE',  # Location of this library on remote host
    XHOST_RESULTS_DIR = '/path/to/results',                  # Directory to store results on remote host
    XHOST_LOG_DIR     = '/path/to/logs'                      # Directory to store logs on remote host
)

result_getter = run_remote(host, name='experiment_001', dataset=kingma_mnist, model=PC, pull=False,
          n_trials=10, num_labeled=100, epochs=10,
          
          LAMBDA     = uniform(1, 100),             # Uniform search space in specified range
          lr         = loguniform(1e-4, 1e-2),      # Logarithmic search space in specified range
          batch_size = integer(50, 200, step=50),   # Uniform search space with quantization
          balance    = categorical([True, False])   # Categorical search space
                              
          )
          
results = result_getter() # Calling the returned object will get the results if they are available
```
