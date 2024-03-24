### Nerf From Scratch

This started as me writing a nerf implimentation from scratch to learn the paper. Eventaully this evolved into adding a set of papers together from modern NERFs into a single repo since I could not find a good easy to use nerf repo that had quality style transfer + works on iphone mp4s

Goal:

Be able to run a single script

```
train.py --video demo.mp4
```

And have a nerf train in 50-60 seconds that allows high quality stylization

Video

- [x] Add a timestep variable
- [x] Enhance sampling for low-quality videos
  - [x] No camera coords or direction required(Learn positon and direction vectors per image)
  - [x] Implement deblurring using a transformer model
  - [x] Prioritize learning from clear frames
  - [x] Increase focus on edges within frames
  - [x] Favor frames with significant changes to minimize redundancy

Models

- [x] Add features from nerf follow up sin+cos
- [x] Explore Transformers(NOTES: Data hungry and slow, is more smooth and better results with more compute)
- [x] Add fast mobile transformer instead of MLP[In progess]
- [x] Lookup table inspired by instantNGP
- [x] Expand lookup table

Style(TODO)

- [ ] CLIP based pixel style loss
- [ ] CLIP Based geometry style loss on depth maps

Training Speedups

- [x] Shoot more rays at edges in images (TODO: Add paper ref)
- [ ] Regularize loss over empty space from InfoNERF
- [ ] Use fast lookup tables nvidia Instant NGP(special cuda kernales)
