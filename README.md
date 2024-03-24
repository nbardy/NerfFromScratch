## Nerf From Scratch

This started as me writing a nerf implimentation from scratch to learn the paper. Eventaully this evolved into adding a set of papers together from modern NERFs into a single repo since I could not find a good easy to use nerf repo that had quality style transfer + works on iphone mp4s

#### Goal:

Be able to run a single script

```
train.py --video demo.mp4
```

Allow training of a video NERF in 50-60 seconds that allows high quality video stylization

#### Models

- [x] Add spacetime lookup tables
- [x] Explore Transformers instead of MLPs(NOTES: Data hungry and slow, is more smooth and better results with more compute)
- [x] Add fast mobile transformer instead of MLP[In progess, code is in repo haven't tested on NERFs)(Probably ditch this in favor of lookup tables)
- [x] Lookup table inspired by instantNGP

#### Expand Spacetime Handling

- [x] Add Space time geometry projection to project arbitrary scene geometry to cube based 3D lookup tables
- [x] Lookup neighbors in lookup table for more scene information in space and time directions

#### Video

- [x] Add a timestep variable
- [x] Enhance sampling for low-quality videos
  - [x] No camera coords or direction required(Learn positon and direction vectors per image)
  - [x] Implement deblurring using a transformer model
  - [x] Prioritize learning from clear frames
  - [x] Increase focus on edges within frames
  - [x] Favor frames with significant changes to minimize redundancy

#### Training Stability

- [x] Add Loss term on model estimated depth, This allows us to take advantage of a prerained foundation model and bootstrap training

#### Style Controls

- [x] CLIP based pixel style loss
- [x] CLIP based geometry style loss on depth maps

#### Training Speedups

- [x] Shoot more rays at edges in images (TODO: Add paper ref)
- [x] Regularize loss over empty space from InfoNERF(ignore KL loss from infoNERF)
- [x] Use fast lookup tables inspired by nvidia Instant NGP
- [] (TODO): Use actual CUDA kernels from instant-NGP(Although these won't be spacetime compatible)(The torch tables may be fast enough)
