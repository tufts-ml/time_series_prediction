# Time Series Prediction

Code for prediction given sequential data with some missing / irregularly-spaced values (PI: Mike Hughes).

## Usual Workflow

```console
$ cd scripts/{dataset}
$ snakemake prep_standardized_data
$ snakemake prep_collapsed_standardized_data
$ snakemake train_lr
$ snakemake train_rnn
$ snakemake train_hmm
```

## Organization

* docs/ : documentation

* datasets/ : Dataset files only
    
    In standardized Tidy CSV format

    Usually, files are TOO LARGE to be in version control.

* scripts/ : Code to prepare datasets and to train+evaluate models

    Organized by dataset.

* src/ : Python source code for core methods










