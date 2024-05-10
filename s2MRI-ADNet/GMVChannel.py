import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from  NFCSChannel import nfsc_channel

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual_basic_block
class residual_basic_block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, change_channel=True):
        super(residual_basic_block, self).__init__()

        self.stride = stride
        self.change_channel = change_channel

        self.conv1 = conv3x3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)

        self.conv2 = conv3x3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        self.change_channel_fuc = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))

    def forward(self, x):
        residual = x

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.change_channel is True:
            residual = self.change_channel_fuc(residual)

        y += residual
        y = self.relu(y)
        return y


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class s2MRI_ADNet(nn.Module):
    def __init__(self, img_shape=(32, 32, 32), input_dim=64, output_dim=3, embed_dim=768, patch_size=4, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 6
        self.ext_layers = [3, 6]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)
        self.nfcs = nfsc_channel()
        self.instance_norm = nn.InstanceNorm3d(1)
        self.conv0 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool0 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = residual_basic_block(32, 32, stride=1, change_channel=False)
        self.layer1_1 = residual_basic_block(32, 32, stride=1, change_channel=False)
        self.avgpool1 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.layer2 = residual_basic_block(32, 64, stride=1, change_channel=True)
        self.layer2_1 = residual_basic_block(64, 64, stride=1, change_channel=False)

        self.layer3 = residual_basic_block(768, 512, stride=1, change_channel=True)
        self.layer3_1 = residual_basic_block(512, 512, stride=1, change_channel=False)
        self.avgpool2 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.layer4 = residual_basic_block(512, 256, stride=1, change_channel=True)
        self.layer4_1 = residual_basic_block(256, 256, stride=1, change_channel=False)
        self.avgpool3 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.GAP = nn.AdaptiveAvgPool3d(1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(352),
            nn.Linear(352, 1)
        )
        self.sigmod =nn.Sigmoid()
    def forward(self, x,graphx,graphedge_index,graphbatch,linear):
        gcnout = self.nfcs(linear,graphx, graphedge_index, graphbatch)
        gcnout1 = gcnout.detach()

        x = self.instance_norm(x)
        x = self.relu(self.bn1(self.conv0(x)))
        x = self.avgpool0(x)

        x = self.layer1_1(self.layer1(x))
        x = self.avgpool1(x)

        x = self.layer2_1(self.layer2(x))


        z = self.transformer(x)

        _, _, z6 = x, *z

        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        x = self.avgpool2(z6)
        x = self.layer3_1(self.layer3(x))
        x = self.avgpool3(x)
        x = self.layer4_1(self.layer4(x))

        x = self.GAP(x)
        x= x.view(x.size(0), -1)

        out = torch.cat([x, gcnout1], dim=1)
        out = self.mlp_head(out)
        out= self.sigmod(out)
        out = out.view(out.size(0))

        return out




