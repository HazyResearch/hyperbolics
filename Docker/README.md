Install CUDA Drivers.
--
 * This assumes you have CUDA 9 installed.
   * `sudo apt-get install cuda-9.0`
   * This may require a reboot

Install Docker
--
  * Install the community edition.
    * https://docs.docker.com/install/linux/docker-ce/ubuntu/
    * You may want this post (https://askubuntu.com/questions/13065/how-do-i-fix-the-gpg-error-no-pubkey/15272#15272)
    * https://stackoverflow.com/questions/13708180/python-dev-installation-error-importerror-no-module-named-apt-pkg/44612200#44612200

  * Add the user to the docker group `sudo usermod -aG docker $USERID`
    * To use without rebooting, add yourself to the group `newgrp docker` as user
    
  * For GPU, Install Nvidia docker for GPU support https://github.com/NVIDIA/nvidia-docker
    * You need to restart docker `service docker stop`
    
    
* Build the docker container using `sh build.sh` in the Docker directory.
* There is a run_docker script (if you change the name of the tag, update this)
  * It runs in interactive omode
