# Docker Image Build Workflow (CUDA 12.9)

This directory contains the GitHub Actions workflow for building vLLM Docker images based on CUDA 12.9.1 and pushing them to GitHub Container Registry (ghcr.io).

## What it does

- Builds the `vllm-openai` target from `docker/Dockerfile` using CUDA 12.9.1
- Pushes the image to `ghcr.io/<owner>/<repo>:<tag>-cu129`
- Supports both tag pushes (`v0.x.y`) and branch pushes (`main`, feature branches)

## Prerequisites

### 1. Self-hosted GPU Runner

The build requires a **self-hosted GitHub Actions runner** with:
- An NVIDIA GPU (any datacenter GPU with SM 7.5+ compute capability)
- NVIDIA drivers installed and working (`nvidia-smi` should show the GPU)
- **NVIDIA Container Toolkit** configured so Docker can access the GPU:
  ```bash
  # Install NVIDIA Container Toolkit (Debian/Ubuntu)
  distribution=$(. /etc/os-release && echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- Docker installed and running
- Sufficient disk space (~50GB+ recommended for the build cache)
- Sufficient RAM (~64GB+ recommended)

### 2. Register the runner with your repository

```bash
# On the self-hosted machine
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token <PAT> --name gpu-runner
./run.sh
```

The PAT needs `repo` and `actions:write` scopes. The runner label `gpu-runner` must match what's in the workflow file.

### 3. Repository Settings

Enable Actions in repository settings:
- Settings → Actions → General → Actions permissions → "Allow actions"

## Triggering the build

### Automatic triggers
- **Push a tag** matching `v*` (e.g., `git push origin v0.6.0`)
- **Push to main branch**

### Manual trigger
- Go to Actions → "Build & Push Docker Image (CUDA 12.9)" → "Run workflow"
- You can customize CUDA version, Ubuntu version, TORCH_CUDA_ARCH_LIST, etc.

## Image tags produced

| Trigger | Tag format |
|---------|-----------|
| Tag push `v0.6.0` | `ghcr.io/<org>/<repo>:0.6.0-cu129` |
| Push to `main` | `ghcr.io/<org>/<repo>:main-cu129` |
| Push to branch `feature-x` | `ghcr.io/<org>/<repo>:feature_x-cu129` |

## Customization

### Change the runner label
Edit `matrix.runner` in `.github/workflows/build-docker-cu129.yml` if your runner uses a different label.

### Add aarch64 support
Add another entry to the `matrix.include` block with an ARM GPU runner.

### Build with KV connectors
Set `INSTALL_KV_CONNECTORS=true` in the environment or via workflow_dispatch input.

## Troubleshooting

### Build fails with "no such device"
Ensure NVIDIA Container Toolkit is properly configured and Docker was restarted after installation.

### Out of disk space
The Docker build context + cache can consume significant space. Run `docker system prune -af` periodically on the runner.

### Pull rate limit from Docker Hub
The build pulls `nvidia/cuda:12.9.1-devel-ubuntu22.04` from Docker Hub. If you hit rate limits, consider mirroring the base image to ghcr.io or a private registry.

### Version detection fails
The workflow uses `setuptools_scm` which relies on git tags. Ensure `fetch-depth: 0` is set (it is by default in this workflow) and that tags are pushed to the remote.
