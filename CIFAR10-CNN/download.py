import sys
import os
import tarfile
import zipfile
import urllib2

# Progress function pass during urlretrieve 
def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

#Download and extract the dataset
#Input:
#   None
#Return:
#   None
def maybe_download_and_extract(data_url, data_path):
    filename = data_url.split('/')[-1]
    file_path = os.path.join(data_path, filename)
    # Check the extraction directory exists, if not make one
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # Check file being extracted or not
    if not os.path.exists(file_path):
        # Retrieve the file from data_url, and save it in filepath
        # For Python 3
        # filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        # For Python 2.7
        print("Fetching data from: " + data_url)
        datafile = urllib2.urlopen(data_url)
        output = open(file_path,'wb')
        output.write(datafile.read())
        output.close()
        print("Extracting data from: " + filename)
        # Check the file type and use corresponding method to decrypt it
        if file_path.endswith(".zip"):
                # Unpack the zip-file.
                zipfile.ZipFile(file=file_path, mode="r").extractall(data_path)
        elif file_path.endswith((".tar.gz", ".tgz")):
                # Unpack the tar-ball.
                tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
    else:
        print("Data has already been extracted during previous session.")