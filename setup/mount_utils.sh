#!/bin/bash

function mount_container {
  if (($# != 5)); then
    echo "ERROR: bad arguments to mount_container()" >&2
    return 1
  fi

  storage_account_name="$1"
  container_name="$2"
  permissions="$3"
  mount_base="$4"
  mount_folder="$5"
  mount_dir="$mount_base/$mount_folder/$container_name/"

  blobfuse_config_file="$HOME/.blobfuse.$storage_account_name.$container_name"

  sudo mkdir -p "$mount_dir" || true  # mkdir -p was actually returning an error even with the -p option, weird

  st=$(stat --file-system --format=%T "$mount_dir")
  if [[ "$st" == "fuseblk" ]]; then
    echo "WARNING: There is already a folder mounted at $mount_dir. Skipping." >&2
    return 0
  fi

  use_adls="false"
  expiry=$(date --date="+1 week" +"%Y-%m-%dT00:00Z")
  if ! sas_token=$(az storage fs generate-sas --account-name "$storage_account_name" --auth-mode login --name "$container_name" --permissions "$permissions" --expiry "$expiry" --as-user --output tsv --only-show-errors); then
    echo "ERROR: Failed to get a storage token for account $storage_account_name. Perhaps you need to 'az login --use-device-code'?" >&2
    return 1
  fi

  cat <<EOF > "$blobfuse_config_file"
accountName $storage_account_name
sasToken $sas_token
authType SAS
containerName $container_name
EOF

  sudo blobfuse \
    --config-file="$blobfuse_config_file" \
    --tmp-path=/mnt/resource/blobfusetmp \
    --use-adls="$use_adls" \
    -o attr_timeout=240 \
    -o entry_timeout=240 \
    -o negative_timeout=120 \
    -o allow_other \
    "$mount_dir"

  echo "$mount_dir mounted with permissions $permissions, SAS token will expire on: $expiry"
}

function unmount_container {
  storage_account_name="$1"
  container_name="$2"
  permissions="$3"
  mount_base="$4"
  mount_dir="$mount_base/$storage_account_name/$container_name/"
  echo Trying to unmount $mount_dir
  if sudo umount -f $mount_dir; then
    echo Folder $mount_dir unmounted.
  else
    echo Unable to unmount $mount_dir, skipping.
  fi
}

export -f mount_container
export -f unmount_container
