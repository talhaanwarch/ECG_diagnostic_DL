from django.shortcuts import render
from .models import image_classification
from django.http import HttpResponseRedirect
from .py_templates.my_model import image_pred
from PIL import Image
import requests
import numpy as np


diseases=['Covid', 'Normal', 'MI', 'MI_recovered', 'Normal']


def home(request):
	
	
	images=image_classification.objects.all()
	try:
		
		url=images[len(images)-1].pic.url
		out=image_pred(url)
		out=list(zip(diseases,out))
		#print('out is',out)
		#out=dis[int(out)]
		print('----------------',url,'----------------')
		print('-----disease is------- ',out,'----------')
		return render(request,'home.html',{'pred':out,'url':url})
	except FileNotFoundError:
		return render(request,'home.html',{'pred':"None"})





def uploadImage(request):
	print('image uploaded via disk')
	img=request.FILES['image']
	image=image_classification(pic=img)
	image.save()
	return HttpResponseRedirect('/')
	#return render(request,'home.html')

def uploadURL(request):
	#file_name='image{}.jpg'.format(np.random.randint())
	print('image is uploaded via url')
	url=request.POST.get('imgurls')
	# img=Image.open(urllib2.urlopen(url))
	# img=Image.open(requests.get(url, stream=True).raw)
	imgurl=requests.get(url, stream=True).raw
	out=image_pred(imgurl)
	out=list(zip(diseases,out*100))
	#out=dis[int(out)]
	#img.save(file_name)
	return render(request,'home.html',{'pred':out,'url':url})
