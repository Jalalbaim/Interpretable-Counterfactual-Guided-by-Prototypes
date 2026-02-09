"""
This a basic implementation of the paper "Intrepretable Counterfactuals Guided by prototypes" (2020)
@author: BAIM M.Jalal
"""

import torch
import torch.optim as optim

class Counterfactuals:
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

    def compute_prototypes(self, x_orig, class_samples, K=5, method=None):
        """
        Compute prototypes for each class based on the samples provided.
        If method is None, use the nearest K samples to the original input.
        If method is 'kmeans', use K the number of clusters.
        """
        if method is None:
            prototypes = {}
            # Move x_orig to the same device as the encoder
            x_orig = x_orig.to(self.device)
            x_orig_enc = self.encoder(x_orig).detach()

            for cls, samples in class_samples.items():
                with torch.no_grad():
                    self.encoder.eval()
                    # Move samples to the same device as the encoder
                    samples = samples.to(self.device)
                    encodings = self.encoder(samples).detach()

                expanded = x_orig_enc.expand_as(encodings)
                dists = torch.norm((encodings - expanded).flatten(1), dim=1)
                nearest_indices = dists.argsort()[:K]
                nearest_encodings = encodings[nearest_indices]
                prototypes[cls] = nearest_encodings.mean(dim=0)

                if K < 10:
                    K += 1

            return prototypes
    # def compute_prototypes(self, x_orig, class_samples, K=5, method=None):
    #     """
    #     Compute prototypes for each class based on the samples provided.
    #     If method is None, use the nearest K samples to the original input.
    #     If method is 'kmeans', use K as the number of clusters.
    #     """
    #     if method is not None:
    #         raise NotImplementedError("KMeans method not implemented yet.")

    #     prototypes = {}
    #     x_orig_enc = self.encoder(x_orig.to(self.device)).detach().to(self.device)

    #     for cls, samples in class_samples.items():
    #         samples_cpu = samples.cpu()
    #         batch_size = 32
    #         encodings_parts = []

    #         for i in range(0, samples_cpu.size(0), batch_size):
    #             batch = samples_cpu[i : i + batch_size].to(self.device)
    #             with torch.no_grad():
    #                 batch = batch.to(self.device)
    #                 self.encoder.eval()
    #                 enc_part = self.encoder(batch)
    #             encodings_parts.append(enc_part.detach().cpu())
    #             torch.cuda.empty_cache()

    #         encodings = torch.cat(encodings_parts, dim=0).to(self.device)

    #         expanded = x_orig_enc.expand_as(encodings)
    #         # Flatten spatial dims if needed
    #         flat_orig = expanded.flatten(1)
    #         flat_samples = encodings.flatten(1)
    #         dists = torch.norm(flat_samples - flat_orig, dim=1)

    #         nearest_idx = dists.argsort()[:K]
    #         nearest_encs = encodings[nearest_idx]
    #         prototypes[cls] = nearest_encs.mean(dim=0)

    #         if K < 101:
    #             K += 1

    #     return prototypes

        
    def algorithm_CGP(self, x_orig, data_tensor,
                      c=1.0,
                      beta=0.01,
                      theta=0.1,
                      cap=0.0,
                      gamma=0.1,
                      K=5,
                      max_iterations=500,
                      lr=1e-2,
                      device=None,
                      writer=None):

        loss_history = []

        # 1 getting preds
        self.model.eval()
        self.autoencoder.eval()


        with torch.no_grad():
            data_preds_list = []
            batch_size = 32 
            for i in range(0, data_tensor.size(0), batch_size):
                batch = data_tensor[i:i + batch_size]
                preds = self.model(batch).argmax(dim=1)
                data_preds_list.append(preds)
            data_preds = torch.cat(data_preds_list).cpu()
            data_tensor = data_tensor.cpu()
            orig_class = self.model(x_orig).argmax(dim=1).item()

        class_samples = {int(cls.item()): data_tensor[data_preds == cls] for cls in torch.unique(data_preds)}

        # 2. Compute prototypes by encoder

        x_orig_enc = self.encoder(x_orig).detach()
        prototypes = self.compute_prototypes(x_orig, class_samples, K)

        # 3 find prototype
        min_dist = float('inf')
        target_class = None
        for cls, proto in prototypes.items():
            if cls != orig_class:
                cur_dist = torch.norm(x_orig_enc - proto)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    target_class = cls
        target_proto = prototypes[target_class]
        print("target class:", target_class)
        #print("target prototype:", target_proto)

        # 4 Optimize the perturbation
        print("Optimizing perturbation...")
        perturbation = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = optim.Adam([perturbation], lr=lr)

        for iter_idx in range(max_iterations):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            cf_candidate = x_orig + perturbation
            cf_candidate = torch.clamp(cf_candidate, 0, 1)

            L_pred = self.loss_pred(cf_candidate, cap, orig_class)
            L1, L2 = self.loss_l1_l2(perturbation)
            L_AE = self.loss_ae(cf_candidate, gamma)
            L_proto = self.loss_proto(cf_candidate, target_proto, theta)
            tot_loss = self.total_Loss(c, L_pred, beta, L1, L2, L_AE, L_proto)
            loss_history.append(tot_loss.item())

            if writer is not None:
                writer.add_scalar('Loss/Total', tot_loss.item(), iter_idx)
                writer.add_scalar('Loss/Prediction', L_pred.item(), iter_idx)
                writer.add_scalar('Loss/L1', L1.item(), iter_idx)
                writer.add_scalar('Loss/L2', L2.item(), iter_idx)
                writer.add_scalar('Loss/AE', L_AE.item(), iter_idx)
                writer.add_scalar('Loss/Proto', L_proto.item(), iter_idx)



            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                cf_pred_class = self.model(cf_candidate).argmax(dim=1).item()
            if cf_pred_class == target_class: #and iter_idx == max_iterations - 1:
                print(f"Counterfactual found at iteration {iter_idx}")
                break


        final_cf = cf_candidate
        return final_cf

