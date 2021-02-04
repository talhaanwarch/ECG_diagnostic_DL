import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np 
from ..apps import SubAppConfig



image_path=os.path.dirname(os.path.dirname(__file__))

aug=transforms.Compose([
                        transforms.Resize((224,224)),
                        #transforms.Grayscale(num_output_channels=1),
                        transforms.RandomHorizontalFlip(1),transforms.RandomVerticalFlip(1),
                        #transforms.RandomPerspective(distortion_scale=0.2),                       
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

                        ])

#load model

#create a function that predict labels
def image_pred(url):
	try:
		new_url=image_path+url
	except TypeError:
		new_url=url
	print('url',new_url)
	img = Image.open(new_url)
	img=img.convert(mode='RGB')
	image = aug(img)
	image=image.unsqueeze(0).cpu() #add another dimension at 0

	SubAppConfig.model.eval()

	out=SubAppConfig.model(image)


	out=torch.mean(out,dim=0)


	out=out.detach().numpy()
	out=np.exp(out)/sum(np.exp(out))

	#out=np.argmax(out)
	return out.round(3)






