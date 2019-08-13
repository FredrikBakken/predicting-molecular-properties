''' Kaggle Download
This script is used to set up the initial directory structure and
download and extract the necessary resource files from Kaggle.

If you're unable to download the Kaggle files, please see
https://github.com/Kaggle/kaggle-api for install guides.
'''

import shutil
import zipfile
from os import mkdir, chdir, listdir
from os.path import exists, isfile, join
from subprocess import Popen, PIPE, STDOUT


def preliminary():
    # Create input directory
    if (exists('../input') == False):
        mkdir('../input')
    
    # Set directory root for download of data sources
    chdir('../input')

    # Download all data source files from Kaggle
    input_cmd = "kaggle competitions download -c champs-scalar-coupling"
    proc = Popen(input_cmd, stdin = PIPE, stdout = PIPE)
    stdout, stderr = proc.communicate(bytes(input_cmd, 'utf-8'))
    print(stdout)

    # Directory to store the zipped original data source files
    if (exists('zipped_source') == False):
        mkdir('zipped_source')

    # Unzip all data source files
    data_source_files = [f for f in listdir('.') if isfile(join('.', f))]
    for data_source_file in data_source_files:
        if ("structures.zip" not in data_source_file):
            zip_ref = zipfile.ZipFile(data_source_file, 'r')
            zip_ref.extractall('.')
            zip_ref.close()

            # Move file to zipped_source
            shutil.move(data_source_file, 'zipped_source/' + data_source_file)


if __name__ == '__main__':
    preliminary()
