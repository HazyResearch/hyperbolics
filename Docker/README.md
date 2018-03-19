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

  * Add the user to the docker group ``sudo usermod -aG docker `whoami` ``
    * To use without rebooting, add yourself to the group `newgrp docker` as user
    
  * For GPU, Install Nvidia docker for GPU support https://github.com/NVIDIA/nvidia-docker
    * You need to restart docker `service docker stop`
    
    
* Build the Docker container using `sh build.sh` in the Docker/ directory.
* Launch the Docker container using `sh run_docker.sh` in the main repository.
  * It runs in interactive mode
