from layer import Layer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
class sequence :
    def __init__(self , num_of_layers ,epoch):
        self.layers = []
        self.num_of_layers = num_of_layers
        self.data  =None

        self.epoch = epoch
    def build_layers(self):
        
        # form GUI take units and activation
        
        self.layers.append(Layer(10, "sigmoid" , 0.0001,5))

        for i in range(1,self.num_of_layers-1):
            previous_layer = self.layers[i-1]
            self.layers.append(Layer(15 , 'sigmoid' ,0.0001, len(previous_layer.a_out)))
        
        previous_layer = self.layers[-1]
        self.layers.append(Layer(3 , 'sigmoid' ,0.0001, len(previous_layer.a_out)))
        




 
                


    def back_propagation(self, target, sample):
        self.layers[-1].error = (target - self.layers[-1].a_out) * self.layers[-1].differentiating
        self.layers[-1].W += self.layers[-1].learning_rate * np.outer(self.layers[-1].error, self.layers[-2].a_out)
        self.layers[-1].bias += self.layers[-1].learning_rate * self.layers[-1].error

        for i in range (len(self.layers)-2,0,-1):
            next_layer=self.layers[i+1]
            current_layer=self.layers[i]
            current_layer.error = np.dot(next_layer.error, next_layer.W) * current_layer.differentiating
            current_layer.W += current_layer.learning_rate * np.outer(current_layer.error, self.layers[i - 1].a_out)
            current_layer.bias += current_layer.learning_rate * current_layer.error

        self.layers[0].error = np.dot(self.layers[1].error, self.layers[1].W) * self.layers[0].differentiating
        self.layers[0].W += self.layers[0].learning_rate * np.outer(list(self.layers[0].error), list(sample))
        self.layers[0].bias += self.layers[0].learning_rate * self.layers[0].error    
        







   

    def forward_propagation(self, sample):
        for neuron in range(self.layers[0].neurons):
            self.layers[0].a_out[neuron]=np.dot(self.layers[0].W[neuron,:],sample)
            self.layers[0].a_out[neuron]=self.layers[0].activation(self.layers[0].a_out[neuron],neuron)+self.layers[0].bias[neuron]
        for i in range(1,len(self.layers)):
            for neuron in range(self.layers[i].neurons):
                self.layers[i].a_out[neuron]=np.dot(self.layers[i].W[neuron,:],self.layers[i-1].a_out)+self.layers[i].bias[neuron]   
                self.layers[i].a_out[neuron]=self.layers[i].activation(self.layers[i].a_out[neuron],neuron)



         
     
                





    def preprocess(self):
      self.data=pd.read_csv('birds.csv')
      gender_distribution = self.data.groupby('bird category')['gender'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
      self.data['gender'] = self.data.apply(lambda row: gender_distribution[row['bird category']] if pd.isnull(row['gender']) else row['gender'], axis=1) 
      label_encoder=preprocessing.LabelEncoder()
      self.data.iloc[:,0]=label_encoder.fit_transform(self.data.iloc[:,0])


      hot_encoder=preprocessing.OneHotEncoder(sparse_output=False)
      encoded_columns = hot_encoder.fit_transform(self.data.iloc[:,-1].values.reshape(-1,1))
      encoded_df = pd.DataFrame(encoded_columns, columns=hot_encoder.get_feature_names_out([self.data.columns[-1]]))
      self.data = pd.concat([self.data.drop(self.data.columns[-1], axis=1), encoded_df], axis=1)
      normalizer=preprocessing.MinMaxScaler()

      #self.data.iloc[:,1:5]=normalizer.fit_transform(self.data.iloc[:,1:5])
      X=self.data.iloc[:,0:5]
      Y=self.data.iloc[:,5:]

      x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,stratify=Y,random_state=42)
      return x_train,x_test,y_train,y_test
    


    def train(self):
        x_train,x_test,y_train,y_test=self.preprocess()
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()


        for j in range(self.epoch):
            for i,sample in enumerate(x_train):
                self.forward_propagation(sample)
                self.back_propagation(y_train[i],sample)




      






