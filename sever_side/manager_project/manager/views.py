from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView

from manager.models import *
from django.core.files.storage import FileSystemStorage
import urllib.parse

def index(request):
    if request.method == 'POST' and request.FILES['upfile']:
        htmlfile = request.FILES['upfile']
        fileobject = FileSystemStorage()
        filedata = fileobject.save(htmlfile.name, htmlfile )
        upload_url = fileobject.url(filedata)
        num_id = PIN.objects.count()
        data = upload_url.strip(".jpg")
        data_fix =data.strip("/media/")
        print(data_fix)
        data_com =data_fix.split("-")
        PIN.objects.create(PINid=num_id+1,Type=data_com[0],rocation_n=data_com[1],rocation_w=data_com[2],data=upload_url)
    data={}
    for data_num in range(PIN.objects.count()):
        data_all = PIN.objects.get(PINid=data_num+1)
        data["data"+str(data_num)]={
            "id":data_all.PINid,
            "Type":data_all.Type,
            "rocation_n":data_all.rocation_n,
            "rocation_w":data_all.rocation_w,
            "data":data_all.data,
        }
    pram ={
        'data':data,
    }
    return render(request,'manager/map.html',pram)
