import torch.nn as nn
import torch
from flask import Flask, render_template, request, redirect, url_for
import pkg_resources

templateFolder = pkg_resources.resource_filename(__name__, 'templates')
app = Flask(__name__, template_folder=templateFolder)

class IncomeClassification(nn.Module):
    def __init__(self):
        super(IncomeClassification, self).__init__()
        #Different types of Activation Function
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=1)
        self.BatchNorm1 = nn.BatchNorm1d(21)
        self.BatchNorm2 = nn.BatchNorm1d(21)
        self.BatchNorm3 = nn.BatchNorm1d(21)
        #hidden layer
        self.hiddenlayer_1 = nn.Linear(7, 21) 
        self.hiddenlayer_2 = nn.Linear(21, 21)
        self.hiddenlayer_3 = nn.Linear(21, 21)
        #output layer
        self.outputlayer = nn.Linear(21, 1) 
        
    def forward(self, input):
        # 1st hidden layer
        x = self.hiddenlayer_1(input)
        x = self.tanh(x)
        x = self.BatchNorm1(x)
        # 2nd Hidden layer
        x = self.hiddenlayer_2(x)
        x = self.sigmoid(x)
        # 3rd Hidden Layer 
        x = self.hiddenlayer_3(x)
        x = self.relu(x)
        # output layer
        x = self.outputlayer(x)
        x = self.sigmoid(x)
        return x

model = IncomeClassification()
model.load_state_dict(torch.load('apriya3_virajmah_assignment2_part1.pt'))
model.eval()

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = dict(request.form)
    input_value_1 = float(data['input-field-1'])
    input_value_2 = float(data['input-field-2'])
    input_value_3 = float(data['input-field-3'])
    input_value_4 = float(data['input-field-4'])
    input_value_5 = float(data['input-field-5'])
    input_value_6 = float(data['input-field-6'])
    input_value_7 = float(data['input-field-7'])
    tensor_input = torch.tensor([input_value_1, input_value_2, input_value_3, input_value_4, input_value_5, input_value_6, input_value_7]).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor_input)
    prediction_value = prediction.tolist()[0]
    return f'The predicted value is {round(prediction_value[0])}'

if __name__ == '__main__':
    app.run()

