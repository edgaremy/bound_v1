import os
import csv  
import wget
from tqdm import tqdm
import asyncio
import sqlite3



def background(f):
	def wrapped(*args, **kwargs):
		return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
		
	return wrapped
	
@background
def get_image(image_url, target_dest):
	wget.download(image_url, target_dest)
	return


if not os.path.exists('Pictures/'):
	os.mkdir('Pictures')


# Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
with open('selected_classes.csv', newline='') as csvfile:
	lines = csvfile.read().split("\n")
	for i,row in enumerate(tqdm(lines)):
		data = row.split(',')
		if i > 0 and len(data) > 2:
			taxon_id = data[0]
			photo_id = data[1]
			extension = data[2]
		
			if not os.path.exists('Pictures/' + taxon_id):
				os.mkdir('Pictures/' + taxon_id)
				
			image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{extension}"
			target_dest = 'Pictures/' + taxon_id + '/' + photo_id + '.' + extension
			get_image(image_url, target_dest)

