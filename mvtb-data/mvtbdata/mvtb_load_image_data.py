#! /usr/bin/env python
from machinevisiontoolbox.base import mvtb_path_to_datafile
from urllib import request, error

# perhaps add a progress bar
# https://stackoverflow.com/questions/41106599/python-3-5-urllib-request-urlopen-progress-bar-available
# using urllib rather than request to minimize number of package installs required

webroot = "https://petercorke.com/files/images/"


def download(filename, force=False):

    # path to where it should be in the mvtb-data package install
    localfile = mvtb_path_to_datafile("images") / filename
    if localfile.exists() and localfile.is_file() and not force:
        print(f"already present as {localfile}")
        return

    # need to get it from the server
    req = request.Request(webroot + filename, headers={"User-Agent": "Mozilla/5.0"})
    try:
        response = request.urlopen(req, timeout=30)
    except error.HTTPError as e:
        print(f"HTTP error {e.code} {e.reason} fetching {webroot + filename}")
        return
    except error.URLError as e:
        print(f"URL error: {e.reason} fetching {webroot + filename}")
        return
    print(f"downloading {filename}...", end="")
    data = response.read()

    # save locally
    f = open(localfile, "wb")
    f.write(data)
    f.close()
    print(f"downloaded {filename} --> {localfile}")


download("bridge-l.zip")
download("bridge-r.zip")
