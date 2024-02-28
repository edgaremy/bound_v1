import os
import csv
import sys  
import wget
from tqdm.auto import tqdm
import asyncio

def get_images_from_inat(src_csv, dest_file, separate_classes_in_folders=False, img_size="original"):
	os.makedirs(dest_file, exist_ok=True)

	def background(f):
		def wrapped(*args, **kwargs):
			return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
			
		return wrapped
		
	@background
	def get_image(image_url, target_dest, pbar):
		wget.download(image_url, target_dest, bar=None)
		pbar.update(1)
		return

	if not os.path.exists(dest_file + 'Pictures/'):
		os.mkdir(dest_file + 'Pictures')


	# Load CSV of selected pictures : #taxon_id	#photo_id #extension #observation_uuid
	with open(src_csv, newline='') as csvfile:
		lines = csvfile.read().split("\n")
		pbar = tqdm(total=len(lines), desc="(ASYNC) INAT SCRAPPING")
		for i,row in enumerate(lines):
			data = row.split(',')
			if i > 0 and len(data) > 2:
				taxon_id = data[0]
				photo_id = data[1]
				extension = data[2]

				if separate_classes_in_folders:

					if not os.path.exists(dest_file + 'Pictures/' + taxon_id):
						os.mkdir(dest_file + 'Pictures/' + taxon_id)
					target_dest = dest_file + 'Pictures/' + taxon_id + '/' + taxon_id + '_' + photo_id + '.' + extension
				else:
					target_dest = dest_file + 'Pictures/' + taxon_id + '_' + photo_id + '.' + extension
				
				image_url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{img_size}.{extension}"
				get_image(image_url, target_dest, pbar)
	
	
	
	
# Usage example:
# csv_number ='6'
# dest_file = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT" + csv_number + "/"
# src_csv = "requested_CSVs/photos_to_scrap_NUMBER" + csv_number + ".csv"
# separate_classes_in_folders = False
# img_size = "original" # "small" (240px)/ "medium" (500px)/ "large" (1024px)/ "original" (2024px)
# get_images_from_inat(src_csv, dest_file, separate_classes_in_folders, img_size)