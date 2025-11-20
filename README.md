# ComfyUI SAM3

ComfyUI custom node pack for **SAM3 (Segment Anything Model 3)** - Meta's state-of-the-art image segmentation model. This extension enables text-prompt-based object segmentation directly within your ComfyUI workflows.

## Overview

SAM3 is a powerful zero-shot segmentation model that can identify and segment objects in images using natural language prompts. This custom node pack brings SAM3's capabilities to ComfyUI, allowing you to:

- Segment objects using text descriptions (e.g., "person", "car", "dog")
- Filter results by confidence threshold
- Control minimum object dimensions
- Output individual masks for each detected object
- Generate a combined mask of all detections
- Visualize results with colored overlays, bounding boxes, and confidence scores

![SAM3 Segmentation Example](screenshot.jpg)

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
2. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
3. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
4. Install SAM3 dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
5. Restart ComfyUI.

## Features

### SAM3 Segmentation Node

**Inputs:**
- **image** - Input image to segment
- **prompt** (STRING) - Text description of objects to segment (e.g., "person", "car", "building")
- **threshold** (FLOAT, 0.0-1.0) - Minimum confidence score threshold for detections (default: 0.5)
- **min_width_pixels** (INT) - Minimum bounding box width in pixels (default: 0)
- **min_height_pixels** (INT) - Minimum bounding box height in pixels (default: 0)

**Outputs:**
- **segmented_image** (IMAGE) - Visualization with colored mask overlays, bounding boxes, and confidence scores
- **masks** (MASK) - Batch of individual binary masks, one for each detected object `[B, H, W]`
- **mask_combined** (MASK) - Single merged mask containing all detected objects `[1, H, W]`

**Example Use Cases:**
- Remove backgrounds by segmenting people or objects
- Isolate specific elements in a scene for further processing
- Create masks for inpainting workflows
- Generate batch masks for multiple objects of the same type
- Filter detections by size to focus on foreground/background objects

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd comfyui_sam3
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/comfyui_sam3/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

