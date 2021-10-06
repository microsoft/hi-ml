# Downloading from/ uploading to Azure ML


## Download files from an Azure ML Run

To download all files from an AML Run, given its run id, perform the following:

```python
from pathlib import Path
from health.azure.azure_util import download_files_from_run_id
download_files_from_run_id("example_run_id_123", Path("path_to_save"))
```

Here, "path_to_save" represents the folder in which we want the downloaded files to be stored.

You can also pass an AML Workspace into this function, or a path to the config file where its settings can be found.
If neither of these is provided, if the function is called during an AML Run, the current workspace will be
used. Otherwise, the code will try to locate a 'config.json' file in your directory.

If you wish to specify the file name(s) to be downloaded, you can do so with the "prefix" parameter. E.g.
prefix="outputs" will download all files within the "output" folder, if such a folder exists within your Run.

There is an additional parameter, "validate_checksum" which defaults to False. If True, will validate 
MD5 hash of the data arriving (in chunks) to that being sent.

Note that if your code is running in a distributed manner, files will only be downloaded onto nodes with local rank = 0.
E.g. if you have 2 nodes each running 4 processes, the file will be downloaded by CPU/GPU 0 on each of the 2 nodes. 

## Downloading checkpoint files from a run

To download checkpoint files from an Azure ML Run, perform the following:

```python
from pathlib import Path
from health.azure.azure_util import download_checkpoints_from_run
download_checkpoints_from_run("example_run_id_123", Path("path/to/checkpoint/directory"))
```

As above, you can pass either an AML Workspace object or a path to a config file to this function, 
or else  if the function is called during an AML Run, the current workspace will be
selected. Otherwise, the code will try to locate a 'config.json' file in your directory.

All files within the checkpoint directory will be downloaded into the folder specified by "path/to/checkpoint_directory".

Since checkpoint files are often large and therefore prone to corruption during download, by default, this function will validate the MD5 hash of the data downloaded (in chunks) compared to that being sent.

Note that if your code is running in a distributed manner, files will only be downloaded onto nodes with local rank = 0.
E.g. if you have 2 nodes each running 4 processes, the file will be downloaded by CPU/GPU 0 on each of the 2 nodes. 


## Downloading files from an Azure ML Datastore

To download data from an Azure ML Datastore within your Workspace, follow this example:
```python
from pathlib import Path
from health.azure.azure_util import download_from_datastore
download_from_datastore("datastore_name", "prefix", Path("path/to/output/directory") )
```

where "prefix" represents the path to the file(s) to be downloaded, relative to the datastore "datastore_name".
Azure will search for files within the Datastore whose paths begin with this string.
If you wish to download multiple files from the same folder, set <prefix> equal to that folder's path
within the Datastore. If you wish to download a single file, include both the path to the folder it
resides in, as well as the filename itself. If the relevant file(s) are found, they will be downloaded to
the folder specified by <output_folder>. If this directory does not already exist, it will be created.
E.g. if your datastore contains the paths ["foo/bar/1.txt", "foo/bar/2.txt"] and you call this
function with file_prefix="foo/bar" and output_folder="outputs", you would end up with the
files ["outputs/foo/bar/1.txt", "outputs/foo/bar/2.txt"]

You can also pass an AML Workspace object, or a path to a config.json file to this function. If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
file will be used to instantiate the workspace.

This function takes additional parameters "overwrite" and "show_progress". If True, overwrite will overwrite any existing local files with the same path. If False and there is a duplicate file, it will skip this file.
If show_progress is set to True, the progress of the file download will be visible in the terminal. 

## Uploading files to an Azure ML Datastore

To upload data to an Azure ML Datastore within your workspace, perform the following:
```python
from pathlib import Path
from health.azure.azure_util import upload_to_datastore
upload_to_datastore("datastore_name", Path("path/to/local/data/folder"), Path("path/to/datastore/folder") )
```

Where "datastore_name" is the name of the registered Datastore within your workspace that you wish to upload to and "path/to/datastore/folder" is the relative path within this Datastore that you wish to upload data to.
Note that the path to local data must be a folder, not a single path. The folder name will not be included in the remote path. E.g. if you specify the local_data_dir="foo/bar"
    and that contains the files ["1.txt", "2.txt"], and you specify the remote_path="baz", you would see the
    following paths uploaded to your Datastore: ["baz/1.txt", "baz/2.txt"]

You can also pass an AML Workspace object, or a path to a config.json file to this function. If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
file will be used to instantiate the workspace.

This function takes additional parameters "overwrite" and "show_progress". If True, overwrite will overwrite any existing remote files with the same path. If False and there is a duplicate file, it will skip this file.
If show_progress is set to True, the progress of the file upload will be visible in the terminal. 