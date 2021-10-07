import urllib.request
import flickrapi
import os

# You need to use your personal credential here. 
# You can get it by registering in Flickr website.

api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
secret_key = "xxxxxxxxxxxxxxxx"

# Flickr api access key
flickr = flickrapi.FlickrAPI(api_key, secret_key, cache=True)

# Names of the classes, used to search images in Flicks
keywords = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
            "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", 
            "potted plant", "sheep", "sofa", "train", "monitor"]

# One folder for each class
folders = [str(i+1).zfill(2) for i in range(len(keywords))]

for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

N = 1000 # number of images to download for each class

# For each class...
for k in range(len(keywords)):

    print(keywords[k])
    i = 0
    for photo in flickr.walk(tag_mode='any',
                             text=keywords[k],
                             extras='url_c',
                             sort='relevance'):
        if i>=N:
            break

        url = photo.get('url_c')
        if url is not None:
            # try block needed since some urls give error and throw an exception
            try:
                save_path = folders[k] + "/" + str(i).zfill(4) + ".jpg"
                urllib.request.urlretrieve(url, save_path)
                i += 1
            except:
                print("Error at url: ", url)





