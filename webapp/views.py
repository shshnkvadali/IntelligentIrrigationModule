from django.shortcuts import render
from django.http import JsonResponse,HttpResponse

# Create your views here.
def predict(request):
    return JsonResponse({'Test':'Success'})

def index(request):
    print("TEST")
    return HttpResponse("<html><body><h1>Use /api/predict for the intelligent irrigation module</h1></body></html>")
