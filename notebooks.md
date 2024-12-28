# Notebooks

## tf_pistachio_model
Mostly scratch, sorted out datasets and transformations. Got a simple model training using the sequential api

## tf_pistachio_model
mostly scratch. Tried getting a model built using the TF model subclassing api, but I wanted the model to be end-to-end, operating on the raw data and with the normalising transformations built in. Getting the normalising layers into the model using subclassing, and adapting them to the data seemed clunky, so I opted for the functional api instead.

## tf_pistachio_funcitonal

Built a simple model using the functional api. This seemed to go well. It was here that I started using mlflow to track experiments, and wanted to start using hyperopt, or some smarter tuning approach for hyperparameter tuning

## tf_pistachio_tuning

Same as above. functional model but using hyperopt to tune. The hyperopt trials object is pickled at the end of each run (and unpickled at the start), so the notebook can just be run multiple times to do multiple trials (5 trials per notebook run)

## tf_pistachio_tuning_fc

Functional model again, but moved the model definition into `pistachio/model.py`. This model is a little more complex
  - variable number of densely connected layers
  - wide and deep architecture (optionally) - raw inputs are also fed into the final layer. 
  - feature crosses (fc) - can provide pairs of features to be crossed. Tried this with a couple of features that ranked poorly in shap analysis, just to see if combining them would add value.

Ran an experiment off this, about 80 trials.

## test evaluation

After running the experiment, and deciding on a model, evaluated this using the test dataset (test set should be used for evaluation, not decision making). Performance was about the same as validation dataset, which should be expected.
