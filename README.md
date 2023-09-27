# deep-learning-challenge
For this code, we create two .ipynb files, one in Juypter Notebook and one using Google Colab for data preprocessing and neural network modelling. 
The first code reads the charity_data.csv file into a Pandas DataFrame and displays the first few rows. 
Then drops non-beneficial ID columns, "EIN" and "NAME." 
The code then determines the unique values in each column and prints the results. 
Then, the code converts categorical data into a numeric format. 
Splits the data into features (X) and target (y) arrays and further into training and testing datasets. 
Scales the data using StandardScaler.
Defines a neural network model with two hidden layers and an output layer. 
Then, the code compiles and trains the model using the training datasets. 
The code evaluates the model's performance on the test data and prints the loss and accuracy. 
For the first code, the value for loss is 2.1972, and the value for accuracy is 0.5338. 
The code ends by saving the trained model to an HDF5 file named "model.h5" 

For the second code written in Colab  
It imports the same necessary libraries and the charity_data.csv file into a Pandas DataFrame and follows a preprocessing route similar to the first code. 
Using LabelEncoder, the code encodes the categorical columns to convert them into numeric values. 
The code then Bins the "ASK_AMT" column into discrete labels based on the specified bins and assigns the result to the "ASK_AMT" column. 
The code follows a similar process for splitting (X) and target (Y) arrays and then scales the feature data using StandardScaler and defines a neural network model with several layers, including the input, hidden and output layers. 
The code compiles the model using the optimizer, loss function, and evaluation metric. 
The model is trained with the training data for 50 epochs and a batch size of 64, with validation data provided. 
The model is evaluated on its performance on the test data and prints the loss and accuracy. 
The loss value is 0.52, and the accuracy value is 0.75
The trained model is saved and moved to a file named "charity_optimized.h5"

Report 
1. The first code preprocesses and cleans a dataset by encoding categorical variables, binning rare categories, splitting the data, scaling features, and then creating a neural network model to predict the success of charity organizations. The second code aims to optimize a neural network model for binary classification utilizing the preprocessed charity dataset. 

2. In both codes, the variable is "IS_SUCCESSFUL" and the target variable for the model. The features for the model are all the other variables aside from "EIN" and "NAME," which were dropped.

3. The second model has three layers: the first layer has 128 neurons, the second layer has 64 neurons, and the third has  32 neurons. The code utilized the ReLU activation function to capture complex patterns. The output layer has 1 neuron with a sigmoid activation function.

   - The code achieved the target model performance of an accuracy score of at least 0.75.
   - The code optimized the model to increase performance by Binning and encoding categorical variables, deep neural architecture with multiple hidden layers and neurons.
   - The model was trained for 50 epochs to allow it to learn over a longer period. 

4. The provided deep learning models successfully preprocess the charity datasets, train them and evaluate their performance. The first code's model performance values were 2.1972 for loss and 0.5338 for accuracy. The second code's model performance values were 0.52 for loss and 0.75 for accuracy, which were much better than the first code's model performance. A lower loss value indicates that the mode's prediction was more in line with actual values, and a higher accuracy value indicates that the model was able to correctly predict the target variable more often. 
