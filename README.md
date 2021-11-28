# Jetson

```bash
$ ./scripts/install_libraries.sh
```

Below are the instructions to build and test the containers using the included Dockerfiles.

## Docker Default Runtime

To enable access to the CUDA compiler (nvcc) during `docker build` operations, add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json` configuration file before attempting to build the containers:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },

  "default-runtime": "nvidia"
}
```

You will then want to restart the Docker service or reboot your system before proceeding.

## Building the Container

To build the containers from a Jetson device running [JetPack 4.6](https://developer.nvidia.com/embedded/jetpack)

```bash
$ git clone https://github.com/LordAlain/Jetson.git
$ cd jetson-containers
```

Before proceeding, make sure you have set your [Docker Default Runtime](#docker-default-runtime) to `nvidia` as shown above.

### ML Containers

To build the ML container (`l4t-ml`), use [`scripts/docker_build_ml.sh`](scripts/docker_build_ml.sh) :

```bash
$ ./scripts/docker_build_ml.sh all        # build all: l4t-pytorch, l4t-tensorflow, and l4t-ml
```

## Testing the Containers

To run a series of automated tests on the packages installed in the containers, run the following from your `Jetson` directory:

```bash
$ ./scripts/docker_test_ml.sh all        # test all: l4t-pytorch, l4t-tensorflow, and l4t-ml
```
