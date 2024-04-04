'''
The entire process of training, generating and evaluating synthetic data is defined 
by the number of models you want to generate. Each generated model concerns a variation 
of the hyperparameters. In our tutorial, we define an amount X of experiments (amount_of_models) 
that will be factored to define the values of the following hyperparameters:

    seq_len: the sequence length would be the size of the temporal window of each sequence used 
             to train the model, that is, how many time steps (lines) each sequence contains.
    hidden_dim: Number of units or neurons in each hidden layer
    batch_size: The batch size determines how many temporal sequences (or how many data examples/lines) 
             are included in a single batch for training.

Thus, the number of models dictates the maximum values for `i`, `j` and `k` within the nested triple loop 
structure used in the scripts. The `fatNum` function will be used to calculate these maximum values. For example, 
if the total number of models created equals 3, then the upper bounds for `i`, `j` and `k` are respectively set 
to 1, 1 and 3. If the number of models is 9, the Maximum values of i`, `j` and `k` are respectively defined as 1, 3 and 3
'''
amount_of_models = 9

# train_steps: Refers to the total number of training iterations
train_steps = 10

# define the number of rows in the synthetic dataset
synth_sample_size=3600

# Number of lines that will be considered to generate statistics
# on both datasets: real and synthetic
statistic_sample_size = 2600


'''
In the synthetic data generation script, we save the data in '.pkl' objects so as not to lose the generated data 
and to not have to re-generate the data every time we want to run the evaluation script. Thus, we define names 
for the objects that will be generated. In this sense, we generate an object to save the real data already 
preprocessed, the synthetic data generated for each model and the statistical metrics of each model that will be 
used to select the best model in the evaluation script.
'''
realdata_obj = 'real3_50_3600_norm'
models_obj = 'models3_50_3600_norm'
metrics_obj = 'metrics3_50_3600_norm'