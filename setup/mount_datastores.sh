#!/bin/bash

source setup/mount_utils.sh
mount_base="/datasetdrive"

# Fill in HIML workspace storage account here before using
STORAGE_ACCOUNT_NAME=""

if [[ -z "$STORAGE_ACCOUNT_NAME" ]]; then
  echo "ERROR: Please fill in the STORAGE_ACCOUNT_NAME variable in setup/mount_datastores.sh" >&2
  return 1
fi

mount_container "" "datasets" "lr" "$mount_base" "himlstorage"
