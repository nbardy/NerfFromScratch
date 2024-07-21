from einops import rearrange, repeat
import torch
from transformers_model_code import TransformerSeq2SeqBasic, TransformerBlock
import torch.nn as nn

import torch
from titok_pytorch import TiTokTokenizer

## Patchify and unpatch

# FROm BxHxWxC to BxTxD  | where B = batch, T = tokens, D = dim
def patchify(image, patch_size):
    x = image

    x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

    return x


# Inverts patchify
# Repeats each feature for each pixel index in the patch
# We add a positional embedding to each pixel and decode to the repeated features for
# each pixel

# B, T, D => B, H, W, D
# Should repeat the D feature for each pixel index in the patch
def unpatch(feature_seq, image_size=None, patch_size=None):
    assert image_size is not None and patch_size is not None, "Both image_size and patch_size must be provided"

    image_h, image_w = image_size

    patch_count = feature_seq.shape[2]
    C_size = 3
    patch_dim = patch_size * patch_size * C_size
    patch_w_count = image_w // patch_size
    patch_h_count = image_h // patch_size

    x = feature_seq

    from einops import repeat

    # token_height, token_width
    th, tw = patch_h_count, patch_w_count

    x = repeat(
        x,
        "b (th tw) d -> b th tw d",
        th=th,
        tw=tw,
    )  # BxHxWxD
    x = repeat(x, "b th tw d -> b (th p1) (tw p2) d", th=th, tw=tw, p1=patch_size, p2=patch_size)

    return x


# assert verbose prints a verbose different message
def assert_shape(x, expected_shape, message="Shape mismatch"):
    if x.shape != expected_shape:
        diff = []
        for index in range(len(expected_shape)):
            # T or F
            if expected_shape[index] != x.shape[index]:
                diff.append(False)
            else:
                diff.append(True)

        # Color encoded highlights errors in x
        diff_string = ""
        for index in range(len(expected_shape)):
            if diff[index] == False:
                diff_string += f"\033[91m{expected_shape[index]}\033[0m"
            else:
                diff_string += f"{expected_shape[index]}"

        message = f"""
        {message}

        expected {expected_shape}
        but got {x.shape}
        """
        raise AssertionError(f"\033[91m{message}\033[0m")


def test_patch_unpatch():
    image = torch.randn(1, 64, 64, 3)  # BxHxWxC
    patch_size = 8
    image_size = [64, 64]
    patches = patchify(image, patch_size)
    assert_shape(patches, (1, 64, 64 * 3), "patches")
    unpatched = unpatch(patches, image_size, patch_size)
    assert_shape(unpatched, (1, 64, 64, 64 * 3), "unpatched")


test_patch_unpatch()

# RMS Norm based on EPS and sqrt root from mistral, doesn't need size of layer or batch
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1)
        return x / norm

#  A small multi layer conv model to project from patch to 

# We used to use a single linear
# self.input_projection_linear = nn.Linear(input_dim * self.patch_size * self.patch_size , inner_dim)
# BxTx(PxPxD) => BxTxD where P=patch size and D=inner dim
# 
# In this class instead we use convultion
# Let's use a single conv1d to project the entire patch to a new patch that is 1/patch_size
# 
class Image2Seq(nn.Module):
    def __init__(self, input_dim, patch_size, inner_dim):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stack_size = patch_size * patch_size * input_dim
        # breaks w and h into patches
        self.patch_1 = lambda x: rearrange(x, 'b (w1 p1) (h1 p2) c -> b w1 h1 p1 p2 c', p1=self.patch_size, p2=self.patch_size)  # Bx(W/ps)(H/ps)x(C*ps*ps)
        # row wise linear rearrange row to last and compress all of the row data to linear
        self.row_rearrange = lambda x: rearrange(x, 'b w h p1 p2 c -> b w h p2 (p1 c)')
        self.col_rearrange = lambda x: rearrange(x, 'b w h p1 p2 c -> b w h p1 (p2 c)')
        self.patch_feature_scale = 2
        self.patch_feature_size = patch_size // self.patch_feature_scale
        self.col_linear = nn.Linear(patch_size * input_dim, self.patch_feature_size)
        self.row_linear = nn.Linear(patch_size * input_dim, self.patch_feature_size)

        # r torch.Size([12, 4, 4, 8, 16]) 
        # c torch.Size([12, 4, 4, 8, 16]) 
        #  Should join this to be 12,4,4,8x16x2
        self.stack_single_rc = lambda x: rearrange(x, 'b w h c r -> b w h (c r)')

        self.gated_ativation = lambda x: torch.cat([x, torch.relu(x)], dim=-1)
        self.stack_all = lambda x, row_x, col_x: torch.cat([x, row_x, col_x], dim=-1)

        # For combining all patch values into one stack for each patch
        self.patch_2 = lambda x: rearrange(x, 'b w h c p1 p2 -> b w h (p1 p2 c)')
        # x 2 for row+col and x2 for relu
        self.final_projection = nn.Linear(self.patch_stack_size + 32 * 2, inner_dim)

        self.flatten = lambda x: rearrange(x, 'b w h c -> b (w h) c')

    def forward(self, x):
        x = self.patch_1(x)

        # Get patch feature via row and col
        r = self.row_linear(self.row_rearrange(x))
        c = self.col_linear(self.col_rearrange(x))
        
        r = self.stack_single_rc(r)
        c = self.stack_single_rc(c)

        # stack
        x = self.patch_2(x)
        x = self.stack_all(x, r, c)
        x = self.final_projection(x)
        x = self.flatten(x)

        return x


# Uses patchify then projects to inner dim then transformer then unpatch then does the final projection to output_dim
class TransformerSeq2SeqImage(nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim=64, heads=8, model_depth=8, memory_tokens = 64, moe=True):
        super().__init__()
        self.patch_size = 8
        # Layer to project pixel patches to inner_dim
        # 1/2 to account for 2x of pos embeddings
        self.input_projection_linear = nn.Linear(input_dim * self.patch_size * self.patch_size , inner_dim)
        # self.input_projection_layer = Image2Seq(input_dim, self.patch_size, inner_dim)
        # Isntead of doing a single full linear, let's do some simple 1D convs that will compress the input by 1/2 each time
        # Single 1 size kernel across the input_dim * patch_size * patch_size
        # Transformer on image tokens

        self.titok = TiTokTokenizer(
            dim = inner_dim,
            num_latent_tokens = 32,   # they claim only 32 tokens needed
            codebook_size = 8192      # codebook size 8192
        )
        self.transformer = TransformerSeq2SeqBasic(inner_dim, inner_dim, inner_dim=inner_dim, heads=heads, model_depth=model_depth, memory_tokens=memory_tokens, moe=moe)



        # reconstructing images from codes


        # Run a small MLP with the input data
        self.final_ff = nn.Linear(inner_dim+input_dim, 12)
        self.gated_activation = lambda x: torch.cat([x, torch.relu(x)], dim=-1)

        # Project inner_dim back to pixels
        self.output_projection_linear = nn.Linear(12 * 2, output_dim)

    
    # Simple embed that adds the pos embeds as sin and cos
    def embed_angles(self, original_x):
        pos_x = original_x[:, :, :, 0]
        pos_y = original_x[:, :, :, 1]

        # [Change] Remapping using sin(x)*cos(x+0.1) for pos_x and sin(y)*cos(x-0.1) for pos_y
        s = torch.sin(pos_x) * torch.cos(pos_x + 0.1)
        c = torch.sin(pos_y) * torch.cos(pos_x - 0.1)

        # Update y and x in place
        # clone with view
        y = original_x.view(original_x.shape)
        y[:, :, :, 0] = s
        y[:, :, :, 1] = c

        return y
    

    def forward(self, x):
        # Save original embeds for final skip connection
        original_x = x

        # x = self.embed_angles(original_x)

        # # For linear we use patchify
        # x = patchify(x, self.patch_size)
        # x = self.input_projection_linear(x)
        print("x", x.shape)
        x = self.titok.tokenize(x)
        x = self.transformer(x)
        x = self.titok.codebook_ids_to_images(x)

        # x = unpatch(x, image_size=original_x.shape[1:3], patch_size=self.patch_size)

        final_embeds = self.embed_angles(original_x)

        # Use final_embeds with residual to merge final features with position for a final 
        # positional skip connection
        # Extract the first two channels and repeat them to match the required dimensions

        # concat final_embeds with x depth wise on C use einops
        x = torch.cat([torch.relu(x), final_embeds], dim=-1)

        x = self.final_ff(x)
        # norm 
        x = self.gated_activation(x)
        x = self.output_projection_linear(x)

        return x 


import torch
import torch.nn as nn

# Outputs a single value classifier
class TransformerClassifyImage(nn.Module):
    def __init__(self, input_dim, inner_dim=64, heads=8, model_depth=8):
        super().__init__()
        self.transformer = TransformerSeq2SeqImage(
            input_dim,
            output_dim=inner_dim,
            inner_dim=inner_dim,
            heads=heads,
            model_depth=model_depth
        )
        self.classifier = nn.Linear(inner_dim, 2)

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        # pool across tokens
        x = x.mean(dim=1)
        # [Change] Use softmax for a one-hot output instead of sigmoid
        # Ensure the output is in one-hot format by using softmax and then converting to one-hot
        x = torch.softmax(x, dim=-1)
        x = torch.round(x)  

        return x


