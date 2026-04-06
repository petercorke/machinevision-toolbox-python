#! /usr/bin/env python
import argparse

from machinevisiontoolbox.base import mvtb_path_to_datafile
from urllib import request, error

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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
    total = response.headers.get("Content-Length")
    total = int(total) if total is not None else None

    if tqdm is None:
        print(f"downloading {filename}...", end="")
    else:
        print(f"downloading {filename}...")

    chunk_size = 64 * 1024

    with open(localfile, "wb") as f:
        if tqdm is None:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        else:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=False,
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"downloaded {filename} --> {localfile}")


def main():
    parser = argparse.ArgumentParser(description="Download MVTB image data files")
    parser.add_argument(
        "--override",
        action="store_true",
        help="download files even if they are already present",
    )
    args = parser.parse_args()

    download("bridge-l.zip", force=args.override)
    download("bridge-r.zip", force=args.override)


if __name__ == "__main__":
    main()
