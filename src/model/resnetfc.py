from typing import List
from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler
import util


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, use_GELU=False, use_BN=False):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.use_BN = use_BN
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if use_GELU:
            self.activation = nn.GELU()
        elif beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()
            
            

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

        if use_BN:
            self.bn_0 = nn.BatchNorm1d(size_in)
            self.bn_1 = nn.BatchNorm1d(size_h)

    def forward(self, x):
        with profiler.record_function("resblock"):
            # v2
            if self.use_BN:
                if x.ndim == 3:
                    net = self.fc_0(self.activation(torch.swapaxes(self.bn_0(torch.swapaxes(x, 1, 2)), 1, 2)))
                    dx = self.fc_1(self.activation(torch.swapaxes(self.bn_1(torch.swapaxes(net, 1, 2)), 1, 2)))
                else:
                    net = self.fc_0(self.activation(self.bn_0(x)))
                    dx = self.fc_1(self.activation(self.bn_1(net)))
            
            # v2.1
            # if self.use_BN:
            #     if x.ndim == 3:
            #        net = self.fc_0(self.activation(torch.swapaxes(self.bn_0(torch.swapaxes(x, 1, 2)), 1, 2)))
            #        dx = self.fc_1(self.activation(torch.swapaxes(self.bn_1(torch.swapaxes(net, 1, 2)), 1, 2)))
            #     else:
            #        net = self.fc_0(self.activation(x))
            #        dx = self.fc_1(self.activation(net))
            
            # v2.2
            # if self.use_BN:
            #     if x.ndim == 3:
            #        net = self.fc_0(self.activation(torch.swapaxes(self.bn_0(torch.swapaxes(x, 1, 2)), 1, 2)))
            #        dx = self.fc_1(self.activation(net))
            #     else:
            #        net = self.fc_0(self.activation(self.bn_0(x)))
            #        dx = self.fc_1(self.activation(net))
            
            else:
                net = self.fc_0(self.activation(x))
                dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx

# Decoder MLP
class DecoderMLP(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out, size_splits, n_blocks, beta=0.0, use_GELU=False, use_BN=False):
        super().__init__()
        # Attributes

        assert size_in == sum(size_splits), 'size_splits must match input tensor size'

        self.size_in = size_in
        self.size_splits = size_splits
        self.size_out = size_out
        self.n_blocks = n_blocks
        self.use_BN = use_BN
        self.use_GELU = use_GELU
        self.n_splits = len(size_splits)

        # Submodules
        self.h_mlps = nn.ModuleList()
        for i in range(self.n_splits):
            self.h_mlps.extend([ResnetBlockFC(size_in=size_splits[i], beta=beta, use_GELU=use_GELU, use_BN=use_BN) for j in range(n_blocks)])
        

    def forward(self, x):
        with profiler.record_function("decoderMLP"):
            spl_num = 0
            splits_out = []
            for spl_id in range(self.n_splits):
                split_x = x[:, :, spl_num : spl_num + self.size_splits[spl_id]]
                spl_num += self.size_splits[spl_id]

                for blk_id in range(self.n_blocks):
                    split_x = self.h_mlps[spl_id * self.n_blocks + blk_id](split_x)
                
                splits_out.append(split_x)

            self.decoder_out = torch.cat(splits_out, dim=-1)
            return self.decoder_out

class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type="average",
        use_spade=False,
        use_GELU=False,
        use_BN=False,
        use_PEcat=False,
        use_sigma_branch=False,
        d_blocks=0,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        if use_sigma_branch:
            d_out = 3
            self.sigma_out = nn.Linear(d_hidden, 1)
            nn.init.constant_(self.sigma_out.bias, 0.0)
            nn.init.kaiming_normal_(self.sigma_out.weight, a=0, mode="fan_in")
        
        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade
        self.use_GELU = use_GELU
        self.use_BN = use_BN
        self.use_PEcat = use_PEcat
        self.use_sigma_branch = use_sigma_branch
        self.d_blocks = d_blocks

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta, use_GELU=use_GELU, use_BN=use_BN) for i in range(n_blocks)]
        )

        if d_blocks > 0:
            self.decoder = DecoderMLP(size_in=d_hidden, size_out=d_hidden, size_splits=[128, 256, 128], n_blocks=d_blocks, beta=beta, use_GELU=use_GELU, use_BN=use_BN)


        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")
            
        if self.use_PEcat:
            self.lin_cat = nn.Linear(self.d_hidden + self.d_in, d_hidden)
            nn.init.constant_(self.lin_cat.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_cat.weight, a=0, mode="fan_in")

        if use_GELU:
            self.activation = nn.GELU()
        elif beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()


    def forward(self, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx
            if self.d_in > 0:
                if self.use_PEcat:
                    x_in = x
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz
                x = self.blocks[blkid](x)

                if blkid == (self.combine_layer - 1):
                    # Add input P.E. feature 在combine layer之前, 用concat(512+42)再經過一層act+FC調整回512
                    if self.use_PEcat:
                        x = torch.cat((x, x_in), dim=-1)
                        x = self.lin_cat(self.activation(x))

                    x = util.combine_interleaved(
                        x, combine_inner_dims, self.combine_type
                    )

                # sb2, combine之後經過2個Block才做sb
                if blkid == (self.combine_layer + 1) and self.use_sigma_branch:
                    sigma = self.sigma_out(self.activation(x))

            if self.d_blocks > 0:
                x = self.decoder(x)

            out = self.lin_out(self.activation(x))

            if self.use_sigma_branch:
                out = torch.cat((out, sigma), dim=-1)
            return out


    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            use_GELU=conf.get_bool("use_GELU", False),
            use_BN=conf.get_bool("use_BN", False),
            use_PEcat=conf.get_bool("use_PEcat", False),
            use_sigma_branch=conf.get_bool("use_sigma_branch", False),
            d_blocks=conf.get_int("d_blocks", 0),
            **kwargs
        )
