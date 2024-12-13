import numpy as np
import matplotlib.pyplot as plt
"""
Designed for PIMA dataset
"""
class MLP:
    def __init__(self, raw):

        raw_data=np.loadtxt(raw)
        self.raw_data=raw_data
        

    def preproceesing(self):
        input = self.raw_data[:, :-1]  
        target = self.raw_data[:, -1] 

        input_min = input.min(axis=0)
        input_max = input.max(axis=0)
        input = (input - input_min) / (input_max - input_min)

        input = np.c_[input,  np.ones(input.shape[0])]  #with bias

        self.input=input
        self.target=target


        self.input_size = input.shape[1]   
        self.hidden_size = 10          
        self.output_size = 1   

        np.random.seed(42)  
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))

        self.learning_rate = 0.1
        self.epochs = 10

    
 
    def training(self):
        for epoch in range(self.epochs):
            for i in range(self.input.shape[0]):

                #Front pass
                self.hidden_input = np.dot(self.input[i], self.weights_input_hidden)
                self.hidden_output = sigmoid_activation(self.hidden_input)

                self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
                self.final_output = sigmoid_activation(self.final_input)

                self.error = self.target[i] - self.final_output


                # Back pass
                self.grad_output = self.error * sigmoid_derivative(self.final_output)
                self.hidden_error = np.dot(self.grad_output, self.weights_hidden_output.T)
                self.grad_hidden = self.hidden_error * sigmoid_derivative(self.hidden_output)

                # Weight update step
                self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_output, self.grad_output)  
                self.weights_input_hidden += self.learning_rate * np.outer(self.input[i],self.grad_hidden)  



    def validation(self):
        self.correct_predictions = []
        for i in range(self.input.shape[0]):
            # Forward pass for each sample
            self.hidden_output = sigmoid_activation(np.dot(self.input[i], self.weights_input_hidden))
            self.final_output = sigmoid_activation(np.dot(self.hidden_output,self.weights_hidden_output))

            self.prediction = 1 if self.final_output >= 0.5 else 0

            if self.prediction == self.target[i]:
                #correct_predictions += 1
                self.correct_predictions.append(self.prediction)
        self.accuracy=len(self.correct_predictions)/self.input.shape[0]
        print(self.accuracy*100)
