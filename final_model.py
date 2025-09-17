import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.Wij = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.Vjk = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))
        self.bias_j = np.random.uniform(-0.5, 0.5, (1, self.hidden_size))
        self.bias_k = np.random.uniform(-0.5, 0.5, (1, self.output_size))
        
        self.custom_accuracy = []
        self.custom_loss = []
        self.epoch_list = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
     
    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_prop(self, x):
        self.hidden_layer_input = np.dot(x, self.Wij) + self.bias_j
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.Vjk) + self.bias_k
        self.output_layer_output = self.softmax(self.output_layer_input)
        
        return self.output_layer_output
    
    def backprop(self, x, y, output):
        error = y - output
        delta_output = error
        
        hidden_layer_error = delta_output.dot(self.Vjk.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        
        self.Vjk += self.hidden_layer_output.T.dot(delta_output) * self.learning_rate
        self.bias_k += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.Wij += x.T.dot(hidden_layer_delta) * self.learning_rate
        self.bias_j += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate
        
                
    def train(self, x, y, epochs, window_size=None):
        for epoch in range(epochs):
            output = self.forward_prop(x)
            self.backprop(x, y, output)
            
            # Calculate accuracy and loss for the current epoch
            correct_count = 0
            total = len(x)
            for i in range(total):
                predicted = np.argmax(self.forward_prop(x[i]))
                if predicted == np.argmax(y[i]):
                    correct_count += 1

            accuracy = (correct_count / total) * 100
            loss = np.mean(np.abs(y - output))
            
            # Append accuracy and loss to lists
            self.custom_accuracy.append(accuracy)
            self.custom_loss.append(loss)
            self.epoch_list.append(epoch)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Accuracy {accuracy:.2f}%, Loss {loss:.4f}')
                
            if window_size and len(self.custom_accuracy) > window_size:
                self.plot_graph(window_size)
    
    def plot_graph(self, window_size):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_list[-window_size:], self.custom_accuracy[-window_size:], label='Accuracy')
        plt.plot(self.epoch_list[-window_size:], self.custom_loss[-window_size:], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Accuracy and Loss')
        plt.legend()
        plt.show()
