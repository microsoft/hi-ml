#!/bin/bash

# Check the ubuntu version
ubuntu_version=$(lsb_release -rs)

# Ubuntu 20.04 is the latest version that includes blobfuse. It works for 22.04 as well, though
if [[ $ubuntu_version == "22.04" ]]; then
    ubuntu_version="20.04"
fi

# Install Linux Software Repository for Microsoft Products
temp_dir=$(mktemp -d)
package_url="https://packages.microsoft.com/config/ubuntu/${ubuntu_version}/packages-microsoft-prod.deb"
wget -P "$temp_dir" "$package_url"
sudo dpkg -i "${temp_dir}/packages-microsoft-prod.deb"

# Update apt-get
sudo apt-get update

# Clean up temporary directory
rm -rf "$temp_dir"
