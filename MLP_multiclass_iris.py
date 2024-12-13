import numpy as np

#activation functions
def sigmoid_activation(weight_sum):
    return 1 / (1 + np.exp(-weight_sum))

def sigmoid_derivative(weight_sum):
    return weight_sum * (1 - weight_sum)


def normal_activation(weight_sum):
    return 1 if weight_sum>0 else 0

class MLP_multiclass:

    def __init__(self, raw):
        raw_data=np.loadtxt(raw)
        self.raw_data=raw_data
        

    def preproceesing(self):
        self.input = self.raw_data[:, :-1]  
        self.target = np.array(self.raw_data[:, -1] , dtype=int)

        self.input_min = self.input.min(axis=0)
        self.input_max = self.input.max(axis=0)
        self.input = (self.input - self.input_min) / (self.input_max - self.input_min)

        self.input = np.c_[self.input,  np.ones(self.input.shape[0])]  #with bias

        self.input_size = self.input.shape[1]  
        self.hidden_size = 10         
        self.output_size = 3  


        np.random.seed(42)  
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))

        self.learn_rate = 0.32
        self.epochs = 100

        #ONE HOT ENCODING is necerrary for target with non binary classes like this
        self.target_one_hot = np.zeros((self.target.size, self.output_size))
        self.target_one_hot[np.arange(self.target.size), self.target] = 1

    def training(self):
        #main training loop 
        for epoch in range(self.epochs):
            for i in range(self.input.shape[0]):

                #Front pass oth layer
                self.hidden_input = np.dot(self.input[i], self.weights_input_hidden)
                self.hidden_output = sigmoid_activation(self.hidden_input)
                # 1st layer
                self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
                self.final_output = sigmoid_activation(self.final_input)

                self.error = self.target_one_hot[i] - self.final_output
                print(self.target_one_hot[i], self.final_output)

                # Back pass from 1st layer to 0th layer
                self.grad_output = self.error * sigmoid_derivative(self.final_output)
                self.hidden_error = np.dot(self.grad_output, self.weights_hidden_output.T)
                self.grad_hidden = self.hidden_error * sigmoid_derivative(self.hidden_output)

                # Weights Update 
                self.weights_hidden_output += self.learn_rate * np.outer(self.hidden_output, self.grad_output)  
                self.weights_input_hidden += self.learn_rate * np.outer(self.input[i], self.grad_hidden)  



    def validation (self):
        self.correct_predict = []
        for i in range(self.input.shape[0]):
            self.hidden_output = sigmoid_activation(np.dot(self.input[i], self.weights_input_hidden))
            self.final_output = sigmoid_activation(np.dot(self.hidden_output, self.weights_hidden_output))
            self.prediction = np.argmax(self.final_output) 
            if self.prediction==self.target[i]:
                self.correct_predict.append(self.prediction)

        self.accuracy= len(self.correct_predict) / self.target.size
        print(self.accuracy*100)

                

                
