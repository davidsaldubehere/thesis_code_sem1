import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import SimplePatchifier, TwoLayerNN


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation

        #I don't see any updatable weights that control aggregation????
        #Like shouldn't it be more like weighted_neighbors = torch.matmul(neibor_features, Wagg) ????

        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        #same here we should have a Wupdate parameter being multiplied ???

        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(
            F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x


class VGNN(nn.Module):
    def __init__(self, in_features=1*16*16, out_feature=320, num_patches=196,
                 num_ViGBlocks=16, num_edges=9, head_num=1):
        super().__init__()

        self.patchifier = SimplePatchifier()
        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])
        
        #all this does is unpacks the list its just a fancy way of turning the list into sequential layers

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class Classifier(nn.Module):

    #in features is channel * patch size * patch size
    def __init__(self, in_features=1*16*16, out_feature=320,
                 num_patches=196, num_ViGBlocks=16, hidden_layer=1024,
                 num_edges=9, head_num=1, n_classes=10):
        super().__init__()
        self.backbone = VGNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num)

        self.predictor = nn.Sequential(
            nn.Linear(out_feature*num_patches, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x

class SegmentationHead(nn.Module):
    def __init__(self, in_features, num_patches, num_classes, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.conv = nn.Conv2d(in_features, num_classes, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(self.num_patches ** 0.5)
        
        # Reshape into the original spatial dimensions
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Up-sample if needed to match the original image resolution
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        
        # Final convolution to output the segmentation map
        x = self.conv(x)

        return x

class SegmentationVGNN(nn.Module):
    def __init__(self, in_features=1*16*16, out_feature=320, num_patches=196,
                 num_ViGBlocks=16, num_edges=15, head_num=1, num_classes=4, patch_size=16):
        super().__init__()

        self.backbone = VGNN(in_features, out_feature, num_patches, num_ViGBlocks, num_edges, head_num)
        
        # Use the segmentation head instead of a classifier
        self.segmentation_head = SegmentationHead(out_feature, num_patches, num_classes, patch_size)

    def forward(self, x):
        features = self.backbone(x)
        segmentation_map = self.segmentation_head(features)
        return segmentation_map

    
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")