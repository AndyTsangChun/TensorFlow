import sys
import os
import tarfile
import zipfile
import urllib2
import subprocess

# Progress function pass during urlretrieve 
def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

# Return the absolute name of the file
def _get_abs_name(name):
    x = name.split('.')
    return '.'.join(x[0:len(x)-1])
    
#Download and extract the dataset
#Input:
#   None
#Return:
#   None
def maybe_download_and_extract(data_url, data_path):
    filename = data_url.split('/')[-1]
    abs_filename = _get_abs_name(filename)
    file_path = os.path.join(data_path, filename)
    # Check the extraction directory exists, if not make one
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # Check file being downloaded or not
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
    else:
        print("Data has already been downloaded in previous session.")
    
    # If the file is .Z convert it to .gz for unpack
    if file_path.endswith((".tar.Z")):
        # Run shell script to uncompress .Z to .gz 
        cmd = 'uncompress ' + data_path + abs_filename + '.Z'
        subprocess.check_output(cmd, shell=True)
        cmd = 'gzip ' + data_path + abs_filename
        subprocess.check_output(cmd, shell=True)
        print("Done converting .Z to .gz")
        
    print("Extracting data from: " + filename)
    # Check the file type and use corresponding method to decrypt it
    if file_path.endswith(".zip"):
        # Unpack the zip-file.
        zipfile.ZipFile(file=file_path, mode="r").extractall(data_path)
    elif file_path.endswith((".tar.gz", ".tgz")):
        # Unpack the tar-ball.
        tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
    elif file_path.endswith((".tar.Z")):
        file_path = os.path.join(data_path, abs_filename + '.gz')
        tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
