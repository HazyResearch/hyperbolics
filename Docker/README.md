Install CUDA Drivers.
--
 * This assumes you have CUDA 9 installed.
   * `sudo apt-get install cuda-9.0`
 

Install Docker
--
  * Install the community edition.
    * https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
    * You may want this post (https://askubuntu.com/questions/13065/how-do-i-fix-the-gpg-error-no-pubkey/15272#15272)
  * For GPU, Install Nvidia docker for GPU support https://github.com/NVIDIA/nvidia-docker
    * You need to restart docker `service docker stop`
    
* Build the docker container using `sh build.sh` in the Docker directory.
* There is a run_docker script (if you change the name of the tag, update this)
  * It runs in interactive omode
