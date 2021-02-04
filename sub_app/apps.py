from django.apps import AppConfig
import os
from django.conf import settings
import torch
class SubAppConfig(AppConfig):
	name = 'sub_app'
	model=model_path = os.path.join(settings.MODELS, 'model.pth')
	model=torch.load(model, map_location=lambda storage, loc: storage)
