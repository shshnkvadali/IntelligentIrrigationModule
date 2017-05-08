from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from sklearn.externals import joblib
from django.views.decorators.csrf import csrf_exempt
import os.path,json,base64,sys, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))

@csrf_exempt 
def predict(request):
    if request.method == 'POST':

        clf = joblib.load(os.path.join(BASE, "model.pkl"))
        with open(os.path.join(BASE, "feature_scale.json"), "r") as data_file:    
            feat_scale = json.load(data_file)
        scaling_vars = [float(var) for var in feat_scale['features_scale'].split(',')]
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        content = body['parameters'].split(' ')
        content = [float(x) for x in content]
        for i in range(0,len(content)):
            content[i]*=scaling_vars[i]
        content = np.array(content).astype(np.float)
        try:
            result = clf.predict(content)
            resp = {'prediction': result[0]}
        except:
            return HttpResponse(sys.exc_info()[0], content_type="application/json")
        return HttpResponse(json.dumps(resp), content_type="application/json")


def index(request):
    return HttpResponse("<html><body><h1>Use /api/predict for the intelligent irrigation module</h1></body></html>")
