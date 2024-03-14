import torch
import torch.nn.functional as F
import torch.nn as nn 


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


class ColorField(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, 6)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        self.mlpWithInputSkip = MLPWithInputSkips(n_layers=6, input_dim=embedding_dim_xyz, output_dim=128, skip_dim=embedding_dim_xyz, hidden_dim=128, input_skips=[3])
        
        self.density_output_layer = nn.Linear(128, 1+128) # no activation


        rgb_output_input_dim = 128

        self.rgb_output_layer = nn.Sequential(
            nn.Linear(rgb_output_input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3), # output RGB
            nn.Sigmoid(),
        )


    def forward(self, sample_points):

        """
        Create a color field that takes in sampled points (vertices of the mesh) and outputs the color of the mesh at those points.
        """
        # the model takes in a RayBundle object in its forward method, and produce color and density for each sample point in the RayBundle.
        xyz_embed = self.harmonic_embedding_xyz(sample_points)
      
        x = self.mlpWithInputSkip(xyz_embed, xyz_embed)
        x = self.density_output_layer(x)
        # x shape [131072, 129], dir_embed shape [131072, 15], sample_directions shape [131072, 3], sample_points shape [131072, 3]
        rgb = self.rgb_output_layer(x[..., 1:])
        return rgb


class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz = 6,
        n_harmonic_functions_dir = 2,
        n_layers_xyz = 6, 
        n_hidden_neurons_xyz = 128,
        append_xyz = [3],
        n_hidden_neurons_dir = 64
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        self.mlpWithInputSkip = MLPWithInputSkips(n_layers=n_layers_xyz, input_dim=embedding_dim_xyz, output_dim=n_hidden_neurons_xyz, skip_dim=embedding_dim_xyz, hidden_dim=n_hidden_neurons_xyz, input_skips=append_xyz)
        
        self.density_output_layer = nn.Linear(n_hidden_neurons_xyz, 1+n_hidden_neurons_xyz) # no activation

        self.view_dependence = True
        rgb_output_input_dim = n_hidden_neurons_xyz+embedding_dim_dir if self.view_dependence else n_hidden_neurons_xyz

        self.rgb_output_layer = nn.Sequential(
            # nn.Linear(256+embedding_dim_dir, 128),
            nn.Linear(rgb_output_input_dim, n_hidden_neurons_dir),
            nn.ReLU(True),
            nn.Linear(n_hidden_neurons_dir, 3), # output RGB
            nn.Sigmoid(),
        )


    def forward(self, ray_bundle):
        # the model takes in a RayBundle object in its forward method, and produce color and density for each sample point in the RayBundle.
        sample_points = ray_bundle.sample_points.view(-1, 3)
        sample_directions = ray_bundle.directions.view(-1, 3) 
        # repeat directions by cfg.n_pts_per_ray times
        n_pts_per_ray = sample_points.shape[0] // sample_directions.shape[0]
        sample_directions = sample_directions.repeat_interleave(n_pts_per_ray, dim=0)
        xyz_embed = self.harmonic_embedding_xyz(sample_points)
        dir_embed = self.harmonic_embedding_dir(sample_directions)
        x = self.mlpWithInputSkip(xyz_embed, xyz_embed)
        x = self.density_output_layer(x)
        density = F.relu(x[..., 0])
    
        if self.view_dependence:
            rgb = self.rgb_output_layer(torch.cat((x[..., 1:], dir_embed), dim=-1))
        else:
            rgb = self.rgb_output_layer(x[..., 1:])
        out = {
            'density': density,
            'feature': rgb
        }

        return out