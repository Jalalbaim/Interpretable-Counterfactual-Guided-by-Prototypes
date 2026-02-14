"""
This a basic implementation of the paper "Intrepretable Counterfactuals Guided by prototypes" (2020)
@author: BAIM M.Jalal
"""

import torch
import torch.optim as optim
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

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

    def _kmeans_torch(self, data, K, num_iters=100, seed=42):
        """
        Minimal PyTorch KMeans fallback for data of shape [N, D].
        Returns centers of shape [K, D].
        """
        generator = torch.Generator(device=data.device)
        generator.manual_seed(seed)

        num_samples = data.shape[0]
        K = min(K, num_samples)
        initial_idx = torch.randperm(num_samples, generator=generator, device=data.device)[:K]
        centers = data[initial_idx].clone()

        for _ in range(num_iters):
            distances = torch.cdist(data, centers)
            assignments = torch.argmin(distances, dim=1)

            updated_centers = []
            for cluster_idx in range(K):
                cluster_points = data[assignments == cluster_idx]
                if cluster_points.numel() == 0:
                    replacement_idx = torch.randint(0, num_samples, (1,), generator=generator, device=data.device).item()
                    updated_centers.append(data[replacement_idx])
                else:
                    updated_centers.append(cluster_points.mean(dim=0))
            new_centers = torch.stack(updated_centers, dim=0)

            if torch.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        return centers

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
        prototypes = {}
        x_orig = x_orig.to(self.device)

        with torch.no_grad():
            self.encoder.eval()
            x_orig_enc = self.encoder(x_orig).detach()
        x_orig_flat = x_orig_enc.flatten(1)

        if method is None:
            for cls, samples in class_samples.items():
                with torch.no_grad():
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

        if method == "kmeans":
            for cls, samples in class_samples.items():
                with torch.no_grad():
                    samples = samples.to(self.device)
                    encodings = self.encoder(samples).detach()

                flat_encodings = encodings.flatten(1)
                num_samples = flat_encodings.shape[0]
                n_clusters = min(K, num_samples)

                if KMeans is not None:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    centers_np = kmeans.fit(flat_encodings.detach().cpu().numpy()).cluster_centers_
                    centers = torch.from_numpy(centers_np).to(flat_encodings.device, dtype=flat_encodings.dtype)
                else:
                    centers = self._kmeans_torch(flat_encodings, n_clusters, seed=42)

                dist_to_orig = torch.norm(centers - x_orig_flat.squeeze(0), dim=1)
                selected_center = centers[dist_to_orig.argmin()]
                prototypes[cls] = selected_center.reshape_as(x_orig_enc.squeeze(0))

            return prototypes

        raise ValueError(f"Unsupported prototype method: {method}")
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
                      beta=0.1,
                      theta=100,
                      cap=0.0,
                      gamma=100,
                      K=5,
                      max_iterations=500,
                      lr=1e-2,
                      proto_method=None,
                      device=None,
                      writer=None,
                      return_details=False):

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
            print("Original class:", orig_class)

        class_samples = {int(cls.item()): data_tensor[data_preds == cls] for cls in torch.unique(data_preds)}

        # 2. Compute prototypes by encoder

        x_orig_enc = self.encoder(x_orig).detach()
        prototypes = self.compute_prototypes(x_orig, class_samples, K=K, method=proto_method)

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

        final_iter = max_iterations - 1
        for iter_idx in tqdm(range(max_iterations)):
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
                    final_iter = iter_idx
                    break


        final_cf = cf_candidate
        if return_details:
            details = {
                "orig_class": orig_class,
                "target_class": target_class,
                "cf_class": self.model(final_cf).argmax(dim=1).item(),
                "final_iteration": final_iter,
                "final_loss": loss_history[-1] if loss_history else None,
                "loss_history": loss_history,
                "prototypes": prototypes,
            }
            return final_cf, details
        return final_cf
