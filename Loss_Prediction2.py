# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:23:27 2019

@author: Pranjall

Assumptions: 1.) Loss to the company's profit occurs when a buyer claims the insurance.
             
             2.) Calculation of 'charges' attribute in the dataset in done based on the health condition, age, policy taken and the premimum.
                 It is assumed to be given and can be calculated using a simple mathematical construct.

Aim: To predict loss.

Intution: A buyer is more likely to claim insurance depending on his personal health, age and possibly even gender, a sample data has thus been created accordingly.
"""

"""_______Applying Polynomial Logistic regression to predict if a buyer claims his insurance or not, and thus incurring loss to the company_______"""

#importing libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#loop to execute the model multiple times on different combinations of training and test set, to calculate average eroor in results.
#This outer most loop can be removed, once the model is fine tuned. (Only to showcase results effectively)
testing_iterations = 100
loop_values = np.zeros((1, testing_iterations))
Actual_loss_values = np.zeros((1, testing_iterations))
Predicted_loss_values = np.zeros((1, testing_iterations))
Execution_times = np.zeros((1, testing_iterations))

for loop in range(0, testing_iterations):
    loop_values[0][loop] = loop + 1

    #monitoring execution time.
    start_time = time.time()
    
    #importing data.
    dataset = pd.read_csv("insurance3r2.csv")
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, 8]
    indices = np.arange(0, len(Y.index))

    #normalizing data.
    from sklearn.preprocessing import StandardScaler
    X_sc = StandardScaler()
    X = X.astype('float')
    X = pd.DataFrame(X_sc.fit_transform(X))
    
    #adding polynomial terms
    from sklearn.preprocessing import PolynomialFeatures
    poly_regressor = PolynomialFeatures(degree = 3)
    X = pd.DataFrame(poly_regressor.fit_transform(X))
    
    #splitting data.
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size = 0.2)
    
    #converting to numpy arrays.
    X = np.array(X_train)
    Y = np.array(Y_train)
    m = len(Y_train)
    Y = Y.reshape(m, 1)        #making size (1003, ) to (1003, 1)
    X_test = np.array(X_test)
    Y_test = (np.array(Y_test)).reshape(len(Y_test), 1)
    
    #initializing parameters.
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1, 1)
    
    #Gradient descent.
    """ These values for learning rate and number of iterations are tuned by observing convergence graphs of the initial linear model (only 8 features) """
    iterations = 250
    learning_rate = 0.7
    parameter_values = np.zeros((9, iterations))
    error_values = np.zeros((1, iterations))
    iteration_values = np.zeros((1, iterations))
    
    for iters in range(iterations):
        
        P = np.dot(X, W) + b
        
        A = 1/(1 + np.exp(-P))
        
        #calculating error.
        """ To save on execution time, regularized equation is not used, model is tuned using trial and error to find the correct amount of polynomial features. """
        E = -(1/m)*(np.dot(Y.T, np.log(A)) + np.dot((1 - Y).T, np.log(1 - A)))
        
        #storing intermediate error values to make graph.
        error_values[0][iters] = E
        
        #calculation of gradient.
        delta_Jw = (1/m)*np.dot((A - Y).T, X)
        delta_Jb = (1/m)*(np.sum(A - Y))
        
        #gradient desccent.
        W = W - learning_rate*(delta_Jw.T)
        b = b - learning_rate*(delta_Jb)
        
        """
        #storing intermediate parameter values to make graph.
        parameter_values[0][iters] = W[0]
        parameter_values[1][iters] = W[1]
        parameter_values[2][iters] = W[2]
        parameter_values[3][iters] = W[3]
        parameter_values[4][iters] = W[4]
        parameter_values[5][iters] = W[5]
        parameter_values[6][iters] = W[6]
        parameter_values[7][iters] = W[7]
        parameter_values[8][iters] = b
        """
    
        #storing intermediate iteration values to make graph.    
        iteration_values[0][iters] = iters

    """
    #plotting the convergence of parameters.
    fig1, ax1 = plt.subplots()   
    ax1.plot(iteration_values[0], parameter_values[0], color = 'red', label = 'W0')
    ax1.plot(iteration_values[0], parameter_values[1], color = 'blue', label = 'W1')
    ax1.plot(iteration_values[0], parameter_values[2], color = 'green', label = 'W2')
    ax1.plot(iteration_values[0], parameter_values[3], color = 'cyan', label = 'W3')
    ax1.plot(iteration_values[0], parameter_values[4], color = 'black', label = 'W4')
    ax1.plot(iteration_values[0], parameter_values[5], color = 'magenta', label = 'W5')
    ax1.plot(iteration_values[0], parameter_values[6], color = 'orange', label = 'W6')
    ax1.plot(iteration_values[0], parameter_values[7], color = 'pink', label = 'W7')
    ax1.plot(iteration_values[0], parameter_values[8], color = 'yellow', label = 'Intercept')
    ax1.set_title('Convergence of Parameters values')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Parameter values')
    ax1.legend()
    
    #plotting the error values.
    fig2, ax2 = plt.subplots() 
    ax2.plot(iteration_values[0], error_values[0], color = 'red')
    ax2.set_title('Convergence of Error vaules')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Error values')
    """
    
    #predicting test values.
    P = np.dot(X_test, W) + b
    Predictions = 1/(1 + np.exp(-P))
    
    for i in range(len(Predictions)):
        if Predictions[i] >= 0.5:
            Predictions[i] = 1
        else:
            Predictions[i] = 0
            
    #making confusion matrix.
    cm = np.zeros((2, 2))
    for i in range(len(Y_test)):
        if Y_test[i] == 1 and Predictions[i] == 1:
            cm[0][0] = cm[0][0] + 1
        elif Y_test[i] == 1 and Predictions[i] == 0:
            cm[0][1] = cm[0][1] + 1
        elif Y_test[i] == 0 and Predictions[i] == 1:
            cm[1][0] = cm[1][0] + 1
        else:
            cm[1][1] = cm[1][1] + 1
    
    #storing execution time.
    execution_time = time.time() - start_time
    Execution_times[0][loop] = execution_time
    
    #displaying the overall performance of the system.
    
    #calculating accuracy.
    diagonal_elements = np.diagonal(cm)
    num = np.sum(diagonal_elements)
    dem = np.sum(cm)        
    accuracy = num/dem
    
    #calculating actual loss.
    charges = np.array(dataset.iloc[:, 7])
    actual_loss = 0
    for i in range(0, len(Y_test)):
        if Y_test[i]:
            actual_loss = actual_loss + charges[indices_test[i]]
            
    Actual_loss_values[0][loop] = actual_loss
            
    #calculating predicted loss.
    predicted_loss = 0
    for i in range(0, len(Predictions)):
        if Predictions[i]:
            predicted_loss = predicted_loss + charges[indices_test[i]]
    
    Predicted_loss_values[0][loop] = predicted_loss
    
fig3, ax3 = plt.subplots()
ax3.plot(loop_values[0], Actual_loss_values[0], c = 'blue', marker = 'o', label = 'Actual loss')
ax3.plot(loop_values[0], Predicted_loss_values[0], c = 'red', marker = 'o', label = 'Predicted loss')
ax3.set_title('Results')
ax3.set_xlabel('Code runs')
ax3.set_ylabel('Loss')
ax3.grid()
ax3.legend()

#displaying the graphs.
plt.show()

""" 
Conclusion: Accuracy of prediction that whether a buyer claims insurance or not is 90%, Thus leading to a prediction error in loss amount of approx +/- 2.5%.
"""

#calculating final results.
Avg_pred_err = np.sum(abs( Actual_loss_values - Predicted_loss_values) / testing_iterations)
Avg_actual_loss =  np.sum(Actual_loss_values) / testing_iterations
print("Average execution time: %f" % (np.sum(Execution_times) / testing_iterations))
print("Average prediction error: %f" % Avg_pred_err)
print("Average actual loss: %f" % Avg_actual_loss)
print("Prediction error in loss amount: %f" % ((Avg_pred_err / Avg_actual_loss) * 100))