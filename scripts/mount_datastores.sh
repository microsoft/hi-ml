#!/bin/bash

source scripts/mount_utils.sh
mount_base="/datasetdrive"

mount_container "himlstoragef191c40dff524" "datasets" "lr" "$mount_base"
