#! /usr/bin/env python
from machinevisiontoolbox.base import mvtb_path_to_datafile
from urllib import request

webroot = 'https://petercorke.com/files/images/'

def download(filename):
    response = request.urlopen(webroot + filename)
    data = response.read()

    localfile = mvtb_path_to_datafile('images', filename)
    f = open(localfile, 'wb')
    f.write(data)
    f.close()
    print(f'downloaded {filename} --> {localfile}')


download('bridge-l.zip')
download('bridge-r.zip')
