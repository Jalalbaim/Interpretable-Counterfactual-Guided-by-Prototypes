import torch
import torch.optim as optim

class Cf_losses:
    def __init__(self, model, encoder, autoencoder=None, device=None):
        """
        model: model that predicts
        encoder: torch.nn.Module
        autoencoder: autoencoder to ensure plausibility results
        """
        self.model = model
        self.encoder = encoder
        self.autoencoder = autoencoder
        self.device = device

    def loss_pred (self, cf_input, cap, original_class):
        """
        L_pred = max( model(cf_input)[original_class] - max_{i ≠ original_class} model(cf_input)[i], -cap )
        """
        output = self.model(cf_input.to(self.device))
        # original logit
        orig_logit = output[:, original_class]

        # max logit among other classes
        num_classes = output.shape[1]
        mask = torch.ones(num_classes, dtype=torch.bool, device=output.device)
        mask[original_class] = False
        max_other_logit = output[:, mask].max(dim=1)[0]

        # pred
        loss = orig_logit - max_other_logit

        # cap
        loss_predict = torch.clamp(loss, min=-cap)

        return loss_predict

    def loss_l1_l2(self, delta):
        """
        - L1 = ||delta||_1
        - L2 = ||delta||_2
        """
        L1 = torch.norm(delta, p=1)
        L2 = torch.norm(delta, p=2)

        return L1, L2

    def loss_ae(self, cf_input, gamma=0.1):
        """
        Verify plausibility par reconstruction
        L_AE = gamma * || x_cf - AE(x_cf) ||_2
        """
        if self.autoencoder is None:
            # If no autoencoder was provided, return 0
            return 0

        x_recon = self.autoencoder(cf_input)
        recon_error = torch.norm(cf_input - x_recon, p=2, dim=1)
        loss = gamma * recon_error.mean()

        return loss
    
    def loss_proto(self, cf_input, proto, theta = 0.1, p=2):
            """
            L_proto = theta * || ENC(x_cf) - proto_j ||_p^2
            """
            encoder_out = self.encoder(cf_input)
            diff = torch.norm(encoder_out - proto, p=p, dim=1)**2
            loss = theta*diff.mean()
            return loss

    def total_Loss(self, c, L_pred, beta, L1, L2, L_AE, Lproto):
        """
        loss = c * L_pred + beta * (L1 + L2) + L_AE + L_proto
        """
        loss = c * L_pred + beta * (L1 + L2) + L_AE + Lproto
        return loss