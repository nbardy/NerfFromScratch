import torch


def dino_pixel_features(image):
    # TODO: Impliment deno model
    return False


def clip_pixel_features(image):
    # TODO: impliment
    return False


def sam_segments(image):
    # TODO: impliment
    return False


def get_feature_map_fake(frame):
    return torch.ones_like(frame)


# REAL
def get_feature_map(frame):
    dino_feature_map = dino_pixel_features(frame)
    pixel_feature_map = clip_pixel_features(frame)
    masks = sam_segments(frame)

    final = torch.zeros_like(frame)

    for mask in masks:
        mask_pixels = mask["mask_pixels"]
        feature_vals = pixel_feature_map * mask_pixels
        # mean across pixels
        pooled = feature_vals.mean(dim=1)
        smoothed_map = pooled * 0.8 + feature_vals * 0.2
        final += smoothed_map

    feature_map = 0.8 * final + 0.2 * dino_feature_map

    return feature_map


def get_video_feature_map(video, cache_key=None):
    if cache_key is None:
        print("[Warning] video features not cached")

    results = []

    for frame in video:
        results.append(get_feature_map(frame))

    # make batch
    results = torch.cat(results, dim=0)

    return results
