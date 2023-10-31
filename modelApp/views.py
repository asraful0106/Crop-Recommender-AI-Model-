from django.shortcuts import render
from joblib import load

model = load('./savedModels/model_rf.joblib')

# Create your views here.

def index(request):
    template = "main/index.html"
    return render(request, template)

def fromInfo(request):
    nitrogen = request.GET['nitrogen']
    phosphorus = request.GET['phosphorus']
    potassium = request.GET['potassium']
    ph = request.GET['ph']
    electricalonductivity = request.GET['electricalonductivity']
    sulphur = request.GET['sulphur']
    copper = request.GET['copper']
    iron = request.GET['iron']
    manganese = request.GET['manganese']
    zinc = request.GET['zinc']
    boron = request.GET['boron']

    # Perform predictions using the model
    input_data = [[nitrogen, phosphorus, potassium, ph, electricalonductivity, sulphur, copper, iron, manganese, zinc, boron]]
    y_pred = model.predict(input_data)
    # specify the output form y_pred
    if y_pred[0] == 0:
        y_pred = 'Grapes'
    elif y_pred[0] == 1:
        y_pred = 'Mango'
    elif y_pred[0] == 2:
        y_pred = 'Mulberry'
    elif y_pred[0] == 3:
        y_pred = 'Pomegrante'
    elif y_pred[0] == 4:
        y_pred = 'Potato'
    elif y_pred[0] == 5:
        y_pred = 'Ragi'
    else:
        y_pred = 'Sorry! Can not predict with this data'
    return render(request, 'main/result.html', {'result' : y_pred})