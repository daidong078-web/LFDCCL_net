import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, deformable=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),


        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = float(drop_rate)

    def forward(self, features):
        concated_features = torch.cat(features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, deformable=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                deformable=deformable
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(1, 1, 1, 1),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2,
                 deformable_blocks=(False, False, False, False)):
        super(DenseNet3D, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=8, stride=8, padding=0, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            #('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            is_deformable = deformable_blocks[i]
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                deformable=is_deformable
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        #out = torch.flatten(out, 1)
        #out = self.classifier(out)
        return out


class VectorVAEHead(nn.Module):
    
    def __init__(self, flattened_dim, latent_dim):
        super().__init__()
        self.fc_shared = nn.Linear(flattened_dim, 512)

        self.fc_mu_s = nn.Linear(512, latent_dim)
        self.fc_logvar_s = nn.Linear(512, latent_dim)

        self.fc_mu_c = nn.Linear(512, latent_dim)
        self.fc_logvar_c = nn.Linear(512, latent_dim)

    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_flat):
        # x_flat: [B, flattened_dim]
        h = F.relu(self.fc_shared(x_flat))

        mu_s = self.fc_mu_s(h)
        logvar_s = self.fc_logvar_s(h)
        mu_c = self.fc_mu_c(h)
        logvar_c = self.fc_logvar_c(h)

        f_s = self.reparameterize(mu_s, logvar_s)
        f_c = self.reparameterize(mu_c, logvar_c)

        return f_s, f_c, mu_s, logvar_s, mu_c, logvar_c


class Decoder3D_V2(nn.Module):
    def __init__(self, latent_dim, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape  # [C, D, H, W], e.g., [128, 4, 5, 4]
        unflattened_dim = unflatten_shape[0] * unflatten_shape[1] * unflatten_shape[2] * unflatten_shape[3]

        self.fc = nn.Linear(latent_dim * 2, unflattened_dim)

        self.refine_convs = nn.Sequential(
            nn.Conv3d(unflatten_shape[0], unflatten_shape[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(unflatten_shape[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(unflatten_shape[0], unflatten_shape[0], kernel_size=3, padding=1)
        )

    def forward(self, f_s, f_c):
        # f_s, f_c: [B, latent_dim]
        x = torch.cat([f_s, f_c], dim=1)  # [B, latent_dim * 2]
        x = F.relu(self.fc(x))

        x = x.view(-1, *self.unflatten_shape)

        reconstructed_z = self.refine_convs(x)

        return reconstructed_z


class Classifier_V2(nn.Module):
    def __init__(self, latent_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, latent_dim * 2]
        logits = self.fc(x)
        return logits


class TemporalDisentanglementModel_V2(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, latent_dim=64, encoder_features=[16, 32, 64, 128],
                 input_shape=(64, 80, 64)):
        super().__init__()

        self.encoder = DenseNet3D(
        growth_rate=32,
        block_config=(1, 1, 1, 1),
        num_init_features=64)
        self.pooling = nn.AdaptiveAvgPool3d(1)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_shape)
            encoder_output = self.encoder(dummy_input)
            self.flattened_dim = encoder_output.numel()
            self.unflatten_shape = encoder_output.shape[1:]  # [C, D, H, W]

        self.vae_head = VectorVAEHead(self.flattened_dim, latent_dim)
        self.decoder = Decoder3D_V2(latent_dim, self.unflatten_shape)
        self.classifier = Classifier_V2(latent_dim, num_classes)

    def forward(self, x_t1, x_t2):
        z_t1 = self.encoder(x_t1)
        z_t2 = self.encoder(x_t2)
        z_t1_pooled = self.pooling(z_t1)
        z_t2_pooled = self.pooling(z_t2)
        z_t1_flat = z_t1.view(z_t1_pooled.size(0), -1)
        z_t2_flat = z_t2.view(z_t2_pooled.size(0), -1)

        f_s_t1, f_c_t1, mu_s_t1, logvar_s_t1, mu_c_t1, logvar_c_t1 = self.vae_head(z_t1_flat)
        f_s_t2, f_c_t2, mu_s_t2, logvar_s_t2, mu_c_t2, logvar_c_t2 = self.vae_head(z_t2_flat)

        f_change = f_c_t2 - f_c_t1
        f_for_classification = torch.cat([f_s_t1,f_s_t2], dim=1)
        #f_for_classification = torch.cat([f_s_t1,f_s_t2], dim=1)
        #f_for_classification = torch.cat([f_c_t1, f_c_t2], dim=1)
        logits = self.classifier(f_for_classification)

        return {
            "logits": logits,
            "f_s_t1": f_s_t1, "f_s_t2": f_s_t2,
            "f_c_t1": f_c_t1, "f_c_t2": f_c_t2,
            "z_t1": z_t1, "z_t2": z_t2,
            "mu_s_t1": mu_s_t1, "logvar_s_t1": logvar_s_t1,
            "mu_c_t1": mu_c_t1, "logvar_c_t1": logvar_c_t1,
            "mu_s_t2": mu_s_t2, "logvar_s_t2": logvar_s_t2,
            "mu_c_t2": mu_c_t2, "logvar_c_t2": logvar_c_t2
        }


def compute_losses_v2(model, model_outputs, labels, temperature=0.05, alpha=0.6, beta=0.3, gamma=0.01, delta=0.0, lam=0.5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor([1, 134/102], dtype=torch.float32).to(device)
    loss_ce = F.cross_entropy(model_outputs["logits"], labels,weight=weights)

    # f_s_t1_norm = F.normalize(model_outputs["f_s_t1"], dim=1)
    # f_s_t2_norm = F.normalize(model_outputs["f_s_t2"], dim=1)
    # sim_matrix = torch.matmul(f_s_t1_norm, f_s_t2_norm.T) / temperature
    # log_prob = F.log_softmax(sim_matrix, dim=1)
    # loss_cl = -log_prob.diag().mean()



    fs1 = F.normalize(model_outputs["f_s_t1"], dim=1)  # [B, D]
    fs2 = F.normalize(model_outputs["f_s_t2"], dim=1)  # [B, D]
    fc1 = F.normalize(model_outputs["f_c_t1"], dim=1)  # [B, D]
    fc2 = F.normalize(model_outputs["f_c_t2"], dim=1)  # [B, D]

    B = fs1.size(0)
    labels_cl = torch.arange(B, device=device)

    logits_12 = (fs1 @ fs2.t()) / temperature                 # [B, B]
    neg_self_1 = torch.sum(fs1 * fc1, dim=1, keepdim=True) / temperature  # [B, 1]

    if lam != 1.0:
        neg_self_1 = neg_self_1 + torch.log(torch.tensor(lam, device=device, dtype=neg_self_1.dtype))

    logits_12_aug = torch.cat([logits_12, neg_self_1], dim=1)  # [B, B+1]
    loss_12 = F.cross_entropy(logits_12_aug, labels_cl)

    logits_21 = (fs2 @ fs1.t()) / temperature                 # [B, B]
    neg_self_2 = torch.sum(fs2 * fc2, dim=1, keepdim=True) / temperature  # [B, 1]

    if lam != 1.0:
        neg_self_2 = neg_self_2 + torch.log(torch.tensor(lam, device=device, dtype=neg_self_2.dtype))

    logits_21_aug = torch.cat([logits_21, neg_self_2], dim=1)  # [B, B+1]
    loss_21 = F.cross_entropy(logits_21_aug, labels_cl)

    loss_cl = 0.5 * (loss_12 + loss_21)




    reconstructed_z_t1 = model.decoder(model_outputs["f_s_t1"], model_outputs["f_c_t1"])
    loss_recon_t1 = F.mse_loss(reconstructed_z_t1, model_outputs["z_t1"].detach())

    reconstructed_z_t2 = model.decoder(model_outputs["f_s_t2"], model_outputs["f_c_t2"])
    loss_recon_t2 = F.mse_loss(reconstructed_z_t2, model_outputs["z_t2"].detach())

    loss_recon = (loss_recon_t1 + loss_recon_t2) / 2

    def kl_divergence(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss_kl_s1 = kl_divergence(model_outputs["mu_s_t1"], model_outputs["logvar_s_t1"])
    loss_kl_c1 = kl_divergence(model_outputs["mu_c_t1"], model_outputs["logvar_c_t1"])
    loss_kl_s2 = kl_divergence(model_outputs["mu_s_t2"], model_outputs["logvar_s_t2"])
    loss_kl_c2 = kl_divergence(model_outputs["mu_c_t2"], model_outputs["logvar_c_t2"])
    loss_kl = (loss_kl_s1 + loss_kl_c1 + loss_kl_s2 + loss_kl_c2) / (4 * model_outputs["logits"].size(0))

    loss_orth_t1 = torch.abs(F.cosine_similarity(model_outputs["f_s_t1"], model_outputs["f_c_t1"])).mean()
    loss_orth_t2 = torch.abs(F.cosine_similarity(model_outputs["f_s_t2"], model_outputs["f_c_t2"])).mean()
    loss_orth = (loss_orth_t1 + loss_orth_t2) / 2
    #print("loss_orth", loss_orth)

    total_loss = loss_ce + alpha * loss_cl + beta * loss_recon + gamma * loss_kl + delta * loss_orth
    return {
        "total_loss": total_loss,
        "loss_ce": loss_ce,
        "loss_cl": loss_cl,
        "loss_recon": loss_recon,
        "loss_kl": loss_kl
    }


if __name__ == '__main__':
    batch_size = 4
    in_channels = 1
    num_classes = 2
    latent_dim = 64
    input_shape = (64, 80, 64)

    model_v2 = TemporalDisentanglementModel_V2(
        in_channels=in_channels,
        num_classes=num_classes,
        latent_dim=latent_dim,
        input_shape=input_shape
    )

    dummy_x_t1 = torch.randn(batch_size, in_channels, *input_shape)
    dummy_x_t2 = torch.randn(batch_size, in_channels, *input_shape)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))


    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v2.to(device)
    dummy_x_t1, dummy_x_t2, dummy_labels = dummy_x_t1.to(device), dummy_x_t2.to(device), dummy_labels.to(device)

    outputs = model_v2(dummy_x_t1, dummy_x_t2)


    losses = compute_losses_v2(model_v2, outputs, dummy_labels)

    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")

    losses['total_loss'].backward()
