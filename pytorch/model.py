import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # tf.LayerNormalization uses eps=0.001
        self.norm_input  = nn.LayerNorm(dim, eps=0.001, elementwise_affine=True)
        self.norm_slots  = nn.LayerNorm(dim, eps=0.001, elementwise_affine=True)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001, elementwise_affine=True)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(b, n_s, -1)
        slots = mu + torch.normal(mean=0, std=1, size=sigma.shape).to(device) * sigma

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1).to(device)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        x = x.reshape((-1, x.shape[-1])).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.decoder_initial_size[0], self.decoder_initial_size[1], 1))

        # Apply decoder
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x
    

class MDSpritesDecoder(nn.Module):
    """Slot Attention paper, Appendix E.3 (page 24)"""
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2).to(device)
        self.conv4 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1).to(device)
        self.decoder_initial_size = resolution
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # Spatial broadcast operation
        x = x.reshape((-1, x.shape[-1])).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.decoder_initial_size[0], self.decoder_initial_size[1], 1))

        # Apply decoder
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        # Split channels
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x


"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, sigmoid=False, mdsprites=True):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.sigmoid = sigmoid

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        if mdsprites:
            self.decoder_cnn = MDSpritesDecoder(self.hid_dim, self.resolution)
        else:
            self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.encoder_layer_norm = nn.LayerNorm(hid_dim, elementwise_affine=True, eps=0.001)
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        #x = nn.LayerNorm(x.shape[1:]).to(device)(x)
        x = self.encoder_layer_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots_rep = self.slot_attention(x)
        # `slots_rep` has shape: [batch_size, num_slots, slot_size].
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots_rep)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, num_channels, width, height].

        if self.sigmoid:
            # Normalize reconstruction to 0-1 range
            recon_combined = torch.nn.functional.sigmoid(recon_combined)

        return recon_combined, recons, masks, slots_rep




class SlotAttentionProjection(SlotAttentionAutoEncoder):
    def __init__(self, resolution, opt, vis=False, mdsprites=True):
        super().__init__(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, sigmoid=opt.bce_loss, mdsprites=mdsprites)

        self.projection_head = ProjectionHead(opt, vis=vis)

    def forward(self, image, vis_step):
        recon_combined, recons, masks, slots = super().forward(image)
        # `recon_combined` has shape: [batch_size, num_channels, width, height].
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].
        # `slots` has shape: [batch_size, num_slots, slot_size].

        if self.training:
            # Only run projection head when training
            return recon_combined, recons, masks, slots, self.projection_head(slots, vis_step)
            # `self.projection_head` returns a dictionary of losses and logged values.

        return recon_combined, recons, masks, slots, None




class ProjectionHead(nn.Module):
    def __init__(self, opt, epsilon=0.0001, vis=False):
        super().__init__()

        self.proj_dim = opt.proj_dim
        self.gamma = opt.std_target
        self.eps = epsilon      # small constant for numerical stability
        self.vis = vis
        self.cov_over_slots = opt.slot_cov

        if opt.cov_div_sq:
            self.cov_div = self.proj_dim**2
        else:
            self.cov_div = self.proj_dim

        # VICReg paper, Section 4.2. Two FC layers with non-linearities and a final linear layer
        norm_layer = None
        if opt.proj_layernorm:
            norm_layer = lambda: nn.LayerNorm(opt.proj_dim)
        else:
            norm_layer = lambda: nn.BatchNorm1d(opt.num_slots)

        if not opt.identity_proj:
            self.projector = nn.Sequential(
                nn.Linear(opt.hid_dim, opt.proj_dim),
                norm_layer(),
                nn.ReLU(),
                nn.Linear(opt.proj_dim, opt.proj_dim),
                norm_layer(),
                nn.ReLU(),
                nn.Linear(opt.proj_dim, opt.proj_dim)
            )
        else:
            assert opt.hid_dim == opt.proj_dim, "Identity projection requires hidden dimension size = projection dimention size"
            self.projector = nn.Identity()


    def forward(self, x, vis_step):
        """
        Calculate projected slot-feature variance and covariance over all the slots for a single batch at once
        TODO write alternative versions of batching described in proposal
        """
        # Given matrix of slot representations, return loss
        projection = self.projector(x)
        # `projection` has shape: [batch_size, num_slots, proj_dim].

        if self.cov_over_slots:
            # Calculate covariance over slots loss for each image separately, then take average. Same for std loss.
            # Take mean over feature dimension for each slot of each batch (separately)
            proj = projection
            proj_mean = torch.mean(proj, dim=2, keepdim=True)
            proj = proj - proj_mean
            proj_batch_sz = proj.shape[2]
            # `proj_mean` has shape: [batch_size, num_slots, 1]
            # `proj` has shape: [batch_size, num_slots, proj_dim]

            # Einstein summation performs per-image matrix multiplication without need for transpose or iteration
            cov = torch.einsum('lij, lkj -> lik', proj, proj) / (proj_batch_sz - 1)
            # `cov_out` has shape: [batch_size, num_slots, num_slots]
            
            # Set all diagonal elements (in each element of the batch) to zero --> equivalent to operating only on off diagonal elements
            cov_calc = cov
            torch.diagonal(cov_calc, 0, dim1=1, dim2=2).zero_()             # isolate off-diag cov. matrix elements for each image
            cov_calc = torch.flatten(cov_calc, start_dim=1, end_dim=2)      # flatten all off-diag elements for each image into a vector
            cov_calc = cov_calc.pow_(2).sum(dim=1).div(self.cov_div)        # apply covariance loss calc. to each image separately
            cov_loss = torch.mean(cov_calc)                                 # average covariance losses over all images in batch
            # `cov_calc` has shape: [batch_size]
            
            # Compare to unoptimized baseline
            # cov_loss_test = sum([self._off_diagonal(cov[idx]).pow_(2).sum().div(self.cov_div) for idx in range(cov.shape[0])]) / cov.shape[0]
            # assert abs(cov_loss_test.item() - cov_loss.item()) < 1e-8
            
            std = torch.sqrt(torch.var(proj, dim=2) + self.eps)
            std_loss = torch.mean(torch.nn.functional.relu(self.gamma - std))    
            # `std` has shape: [num_slots]
        else:
            # Calculate covariance loss over projected slot features
            # Collect all slots over the entire batch
            proj = projection.reshape((-1,) + projection.shape[2:])
            # `proj` has shape: [batch_size*num_slots, proj_dim].
            
            # Our "batch size" here is: (batch size) x (num_slots)
            # NOTE: previously, this was x.shape[0] which is actually equal to batch size
            # when we expected it to be (batch size) x (num_slots). Luckily, num_slots
            # has been constant over all runs so we will just need to adjust cov_weight by 
            # a factor of num_slots to replicate past results for cov over slot features.
            proj_batch_sz = proj.shape[0]

            # Zero-center each projection dimension (subtract per-dimension mean)
            proj_mean = torch.mean(proj, dim=0)
            proj = proj - proj_mean
            # `proj_mean` has shape: [proj_dim].
            # `proj` has shape: [batch_size*num_slots, proj_dim].

            cov = (proj.T @ proj) / (proj_batch_sz - 1)
            cov_loss = self._off_diagonal(cov).pow_(2).sum().div(self.cov_div)
            # `cov` has shape: [proj_dim, proj_dim].

            # Calculate variance loss over projected slot features
            std = torch.sqrt(torch.var(proj, dim=0) + self.eps)
            std_loss = torch.mean(torch.nn.functional.relu(self.gamma - std))    
            # `std` has shape: [proj_dim].
            # cov. over slots shape: [num_slots].

        out = {"std_loss": std_loss, "cov_loss": cov_loss, 'proj_mean': torch.mean(proj_mean).item(),
               'proj_batch_sz': proj_batch_sz}

        if self.vis:
            out['cov_mx'] = cov.detach().cpu()
            out['std_vec'] = std.detach().cpu()

        if vis_step:
            # Take vector norm of the representation and projection for each slot in the batch
            proj_vec_norm = torch.linalg.vector_norm(projection, dim=2)
            proj_input_norm = torch.linalg.vector_norm(x, dim=2)
            # `proj_vec_norm` has shape: [batch_size, num_slots].
            # `proj_input_norm` has shape: [batch_size, num_slots].

            # Broad-brush view of projection head expressive power, take mean of projection/input norms 
            out['proj_out_norm'] = torch.mean(proj_vec_norm).item()
            out['proj_input_norm'] = torch.mean(proj_input_norm).item()

            # Take difference of projection and input norms for each slot, then take the mean of this difference
            # Finer-grained POV of how individual vectors are stretched by projection head
            out['proj_diff_norm'] = torch.mean(proj_vec_norm - proj_input_norm).item()

        return out
    
    def _off_diagonal(self, x):
        # Source: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L239
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




class MDSpritesFeaturePrediction(nn.Module):
    def __init__(self, opt, resolution):
        super().__init__()

        # Load and freeze pretrained model
        if opt.base:
            self.frozen = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim, sigmoid=False, mdsprites=True).to(device)
        else:
            self.frozen = SlotAttentionProjection(resolution, opt, vis=False, mdsprites=True).to(device)
        ckpt = torch.load(opt.model_ckpt, map_location=device)
        self.frozen.load_state_dict(ckpt['model_state_dict'])

        # Freeze pretrained model
        for param in self.frozen.parameters():
            param.requires_grad = False

        # One hidden dimension MLP (performs just as well as MLP with more layers, https://arxiv.org/pdf/2107.00637.pdf Figure 14)
        self.prediction_head = nn.Sequential(
            nn.Linear(opt.hid_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 10),       # 3xcolor, scale, x, y, one-hot encoded shape (MISC?, ellipse, heart, square)
        )
        self.base = opt.base


    def forward(self, x):
        if self.base:
            recon_combined, recons, masks, x, _ = self.frozen(x)
        else:
            recon_combined, recons, masks, x, _ = self.frozen(x, False)
        return self.prediction_head(x), recon_combined, recons, masks


