# LT2222 V23 Assignment 3

# <b>Documentation</b>

1. a3_features

When running the "a3_features.py" file, there are following command line arguments.

* "inputdir", which takes on the input directory where all the data (mails) is located
* "outputfile", which takes on the name of the while where the output table will be saved; please make sure to follow the name with ".csv" to facilitate later use
* "dims", takes on the number of the feature dimensions used in the output table
* "--test" takes on the percentage of the data that will be labelled as test; this argument is optional and if not used, the default is set to 20 

To provide an example, let's say that the data is located in the folder called "enron_sample" that is saved on the Desktop, while the current working directory is set on the Desktop, and "a3_features.py" is also located there. The user wishes to save the table under the name "table", have 100 feature dimensions, and the test portion of 30%. This can be run as follows:

```python3 a3_features.py enron_sample table.csv 100 --test 30```
  
  Please note that if the user was lacking the "--test" argument, the code would still run but the test portion would be set to 20%.
  
 2. a3_model
 
 When running the "a3_model.py" file, there are following command line arguments.
 
 * "featurefile" takes on the file with the csv table that was generated by a3_features
 * "--hidden_size" takes on the size of the hidden layer; this argument is optional and if not used, the default is set to 0
 * "--nonlinear" takes on the type of the non-linearity function the user wishes to use; one can choose between "relu", "tanh", "none"; this argument is optional and if not used, the default is set to "none"
 
To provide an example, let's say that the user wants to take the previously created "table.csv" file, set 5 as the size of the hidden layer, and have the non-linearity function set to "tanh". This can be run as followws:
 
```python3 a3_model.py table.csv --hidden_size 5 --nonlinear "tanh"```

# <b> Questions </b>

## Enron data ethics

Although the dataset has proven to be extremely useful in machine learning research, some ethical concerns cannot be overlooked. In particular, when stripping the signature lines in "a3_features" file, I took on the approach of browsing through the emails (although haphazardly) and trying to eliminate as many signatures as possible. That made me take a peak into the contents of the emails. Aside from the obvious ethical breach that is disclosing all the personal data of the people involved, such as their addresses, phone numbers, etc. I have also noticed that personal matters as well as beliefs were to be found in them. For example, one of the emails involved an invitation to participate in a religious event. That directly exposed the religious beliefs of the author, which is considered sensitive data. Another ethical concern I can think of that the emails also disclose information about individuals not working at the company - I have also came across emails exchanged with journalists or advisors (as well as, of course, customers) not affiliated directly with the company. Their data, however, still became publicly available. Although the leak of personal information seems unavoidable in such a case, I wonder if some of the sensitive information could have been encrypted at least. Maybe it was not possible back then with such a huge amount of data but now, with the development of the artificial intelligence, maybe such a task could be entrusted to an AI. Another idea might also be using pseudonyms instead of the actual names of the employees.

## Results

According to my observations, the greater the number of the hidden layers, the better the results, as observed by the confusion matrices. However, alternating between different types of the nonlinearity functions does not seem to weigh on the results greatly. However, between the two nonlinearity models, I think that <i> Tanh</i> results in a slightly better performance than <i>ReLU</i>, but that also varies depending on the size of the hidden layers.
