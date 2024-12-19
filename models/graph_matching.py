import torch
import torch.nn.functional as F
from torch import nn
import sklearn.cluster as cluster

from echonet.models.transformer import MultiHeadAttentionWrapper
from echonet.models.affinity_layer import Affinity
import numpy as np

INF = 100000000


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha

        loss = -alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
            1 - alpha
        ) * pt**self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class GModuleSelfAttention(torch.nn.Module):
    def __init__(self, in_channels, num_classes, device):
        super(GModuleSelfAttention, self).__init__()
        init_item = []
        self.device = device
        self.num_classes = num_classes
        self.matching_loss_type = "FL"
        self.with_cluster_update = True  # add spectral clustering to update seeds
        self.with_intra_domain_graph = True
        channels_in = 512
        channels_out = 256
        input_dim = 256
        hidden_dim = 128

        self.head_in_ln = nn.Sequential(
            nn.Linear(
                channels_in, channels_out
            ),  # Linear transformation on the last dimension
            nn.LayerNorm(
                channels_out, elementwise_affine=False
            ),  # Normalize across the feature dimension
            nn.ReLU(),  # Activation
            nn.Linear(256, 256),
            nn.LayerNorm(256, elementwise_affine=False),
        )

        init_item.append("head_in_ln")
        # Define the MLP using Sequential
        self.node_cls = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, self.num_classes),  # Hidden to output layer
        )
        self.linear = nn.Linear(512, 256)

        # self.matching_loss = BCEFocalLoss()
        init_item.append("node_cls_middle")

        # Graph-guided Memory Bank
        self.seed_project_left = nn.Linear(
            256, 256
        )  # projection layer for the node completion
        self.register_buffer(
            "sr_seed", torch.randn(self.num_classes, 256)
        )  # seed = bank
        self.register_buffer("tg_seed", torch.randn(self.num_classes, 256))

        # Initialize attention wrapper
        self.intra_domain_graph = MultiHeadAttentionWrapper()
        self.cross_domain_graph = MultiHeadAttentionWrapper()  # Cross Graph Interaction

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=256)
        self.InstNorm_layer = nn.InstanceNorm2d(1)

        # Loss function for node classification

        if self.matching_loss_type == "L1":
            self.matching_loss = nn.L1Loss(reduction="sum")
        elif self.matching_loss_type == "MSE":
            self.matching_loss = nn.MSELoss(reduction="sum")
        elif self.matching_loss_type == "FL":
            self.matching_loss = BCEFocalLoss()

        self.node_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        # Initialize weights
        self._init_weights(init_item)

    def _init_weights(self, init_item):
        nn.init.normal_(self.seed_project_left.weight, std=0.01)
        nn.init.constant_(self.seed_project_left.bias, 0)
        for layer in self.node_cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

        # Initialize weights for `head_in_ln`
        for layer in self.head_in_ln:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(
        self, features_s, score_maps_s=None, features_t=None, score_maps_t=None
    ):
        """
        Forward pass for source domain features and labels.
        """

        num_samples = 64
        middle_head_loss = {}

        nodes_1, labels_1 = self.sample_points(features_s, score_maps_s, num_samples)

        nodes_2, labels_2 = self.sample_points(features_t, score_maps_t, num_samples)
        if nodes_1 is None:
            return None, None, {}

        nodes_1 = self.head_in_ln(nodes_1)
        nodes_2 = self.head_in_ln(nodes_2)
        nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)

        nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

        self.update_seed(nodes_1, labels_1, nodes_2, labels_2)

        nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)

        # compute loss based on samples points
        loss_cls_1 = self._compute_node_loss(nodes_1, labels_1)
        loss_cls_2 = self._compute_node_loss(nodes_2, labels_2)
        loss_cls = loss_cls_1 + loss_cls_2

        middle_head_loss.update({"node_loss": 1.0 * loss_cls})

        # Refine graph nodes with self-attention and Sinkhorn optimization
        loss_sinkhorn, M = self._refine_graph_with_sinkhorn(
            nodes_1, labels_1, nodes_2, labels_2
        )
        middle_head_loss.update({"mat_loss_aff": 10.0 * loss_sinkhorn})

        return nodes_1, middle_head_loss

    def sample_points(self, encoder_feats, decoder_feats, num_samples):
        batch_size, encoder_channels, height, width = encoder_feats[0].shape
        decoder_feats = decoder_feats[0]
        _, decoder_channels, _, _ = decoder_feats.shape

        # Adjust decoder channels to match encoder channels if necessary
        if decoder_channels != encoder_channels:
            conv1x1 = nn.Conv2d(decoder_channels, encoder_channels, kernel_size=1).to(
                self.device
            )
            decoder_feats = decoder_feats.to(self.device).float()
            decoder_feats = conv1x1(decoder_feats)

        # Apply sigmoid activation to decoder outputs
        sigmoid_decoder = torch.sigmoid(decoder_feats)

        # Find valid spatial indices where values exceed the threshold
        valid_indices = (sigmoid_decoder > 0.5).nonzero(
            as_tuple=False
        )  # Shape: [N_valid, 4] (batch, channel, h, w)

        # Sample points from valid (positive) and invalid (negative) indices
        positive_indices = valid_indices
        negative_indices = (sigmoid_decoder <= 0.5).nonzero(as_tuple=False)

        sampled_encoder = []
        sampled_decoder = []

        for b in range(batch_size):
            # Filter indices for the current batch (positive and negative samples)
            pos_indices_batch = positive_indices[positive_indices[:, 0] == b][
                :, 2:
            ]  # Select spatial indices (h, w) for positive
            neg_indices_batch = negative_indices[negative_indices[:, 0] == b][
                :, 2:
            ]  # Select spatial indices (h, w) for negative

            # Sample from positive and negative indices
            sampled_indices_pos = pos_indices_batch
            sampled_indices_neg = neg_indices_batch

            # If there are more valid (positive) indices than needed, randomly sample
            if pos_indices_batch.size(0) > num_samples:
                sampled_indices_pos = pos_indices_batch[
                    torch.randint(0, pos_indices_batch.size(0), (num_samples,))
                ]

            # If there are more invalid (negative) indices than needed, randomly sample
            if neg_indices_batch.size(0) > num_samples:
                sampled_indices_neg = neg_indices_batch[
                    torch.randint(0, neg_indices_batch.size(0), (num_samples,))
                ]

            # Sample encoder features for positive and negative
            encoder_sample_pos = encoder_feats[0][
                b, :, sampled_indices_pos[:, 0], sampled_indices_pos[:, 1]
            ].T  # (num_samples, Channels)
            encoder_sample_neg = encoder_feats[0][
                b, :, sampled_indices_neg[:, 0], sampled_indices_neg[:, 1]
            ].T  # (num_samples, Channels)

            # Set decoder features for positive to 1 and negative to 0
            decoder_sample_pos = torch.ones(
                (sampled_indices_pos.size(0), decoder_channels), device=self.device
            )  # Positive samples set to 1
            decoder_sample_neg = torch.zeros(
                (sampled_indices_neg.size(0), decoder_channels), device=self.device
            )  # Negative samples set to 0

            # Append positive and negative samples
            sampled_encoder.append(
                torch.cat([encoder_sample_pos, encoder_sample_neg], dim=0)
            )
            sampled_decoder.append(
                torch.cat([decoder_sample_pos, decoder_sample_neg], dim=0)
            )

        return torch.stack(sampled_encoder), torch.stack(sampled_decoder)

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):
        k = 20  # Minimum nodes required for clustering
        batch_size, num_samples, channels = sr_nodes.shape

        # Flatten the batch and num_samples dimensions for easier processing
        sr_nodes = sr_nodes.view(
            -1, channels
        )  # Shape: (batch_size * num_samples, channels)
        sr_labels = sr_labels.view(-1)  # Shape: (batch_size * num_samples,)
        # Ensure sr_seed has two entries (one for each class, 0 and 1)
        if self.sr_seed.shape[0] < 2:
            self.sr_seed = torch.zeros(
                2, channels
            )  # Initialize with zeros or appropriate values

        for cls in sr_labels.unique().long():
            # Select the nodes belonging to the current class
            bs = sr_nodes[sr_labels == cls].detach()
            if len(bs) > k and self.with_cluster_update:
                device = "cpu"  # Get the device of bs
                seed_cls = self.sr_seed[cls].to(
                    device
                )  # Move seed_cls to the same device as bs

                # Perform spectral clustering
                sp = cluster.SpectralClustering(
                    2,
                    affinity="nearest_neighbors",
                    n_jobs=-1,
                    assign_labels="kmeans",
                    random_state=1234,
                    n_neighbors=len(bs) // 2,
                )

                # Concatenate seed and selected nodes, then cluster
                bs = bs.to(device)
                indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())

                indx = (indx == indx[0])[
                    1:
                ]  # Remove the first index (seed_cls) from the clustering result
                bs = bs[indx].mean(
                    0
                )  # Update the class seed using mean of clustered nodes
            else:
                bs = bs.mean(0)  # If not enough nodes, just take the mean

            # Update the seed using momentum and cosine similarity
            momentum = torch.nn.functional.cosine_similarity(
                bs.unsqueeze(0).to(device), self.sr_seed[cls].unsqueeze(0).to(device)
            )
            momentum = momentum.to(device)
            self.sr_seed[cls] = self.sr_seed[cls].to(device)
            self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)

        if tg_nodes is not None:
            # Flatten and process target nodes/labels similarly
            tg_nodes = tg_nodes.view(
                -1, channels
            )  # Shape: (batch_size * num_samples, channels)
            tg_labels = tg_labels.view(-1)  # Shape: (batch_size * num_samples,)

            # Ensure tg_seed has two entries (one for each class, 0 and 1)
            if self.tg_seed.shape[0] < 2:
                self.tg_seed = torch.zeros(
                    2, channels
                )  # Initialize with zeros or appropriate values

            for cls in tg_labels.unique().long():
                bs = tg_nodes[tg_labels == cls].detach()

                if len(bs) > k and self.with_cluster_update:
                    # Ensure both bs and seed_cls are on the same device
                    device = "cpu"  # Get the device of bs
                    seed_cls = self.tg_seed[cls].to(
                        device
                    )  # Move seed_cls to the same device as bs
                    sp = cluster.SpectralClustering(
                        2,
                        affinity="nearest_neighbors",
                        n_jobs=-1,
                        assign_labels="kmeans",
                        random_state=1234,
                        n_neighbors=len(bs) // 2,
                    )
                    seed_cls = self.tg_seed[cls]
                    bs = bs.to(device)
                    indx = sp.fit_predict(
                        torch.cat([seed_cls[None, :], bs]).cpu().numpy()
                    )
                    indx = (indx == indx[0])[1:]
                    bs = bs[indx].mean(0)
                else:
                    bs = bs.mean(0)

                momentum = torch.nn.functional.cosine_similarity(
                    bs.unsqueeze(0).to(device),
                    self.tg_seed[cls].unsqueeze(0).to(device),
                )
                momentum = momentum.to(device)
                self.tg_seed[cls] = self.tg_seed[cls].to(device)
                self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def _forward_preprocessing_source_target(
        self, nodes_1, labels_1, nodes_2, labels_2
    ):
        """
        nodes: sampled raw source/target nodes
        labels: the ground-truth/pseudo-label of sampled source/target nodes
        weights: the confidence of sampled source/target nodes ([0.0,1.0] scores for target nodes and 1.0 for source nodes )

        We permute graph nodes according to the class from 1 to K and complete the missing class.

        """

        sr_nodes, tg_nodes = nodes_1, nodes_2
        sr_nodes_label, tg_nodes_label = labels_1, labels_2
        labels_exist = torch.cat([sr_nodes_label, tg_nodes_label]).unique()

        sr_nodes_category_first = []
        tg_nodes_category_first = []

        sr_labels_category_first = []
        tg_labels_category_first = []

        sr_weight_category_first = []
        tg_weight_category_first = []

        for c in labels_exist:

            sr_indx = (sr_nodes_label == c).expand_as(sr_nodes)
            tg_indx = (tg_nodes_label == c).expand_as(tg_nodes)

            sr_nodes_c = sr_nodes[sr_indx]
            tg_nodes_c = tg_nodes[tg_indx]

            if (
                sr_indx.any() and tg_indx.any()
            ):  # If the category appear in both domains, we directly collect them!

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                labels_sr = sr_nodes_c.new_ones(len(sr_nodes_c)) * c
                labels_tg = tg_nodes_c.new_ones(len(tg_nodes_c)) * c

                sr_labels_category_first.append(labels_sr)
                tg_labels_category_first.append(labels_tg)

            elif (
                tg_indx.any()
            ):  # If there're no source nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(tg_nodes_c)
                sr_nodes_c = (
                    self.sr_seed[int(c.item())].unsqueeze(0).expand(num_nodes, 256)
                )

                if self.with_semantic_completion:
                    sr_nodes_c = (
                        torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device)
                        + sr_nodes_c
                        if len(tg_nodes_c) < 5
                        else torch.normal(
                            mean=sr_nodes_c,
                            std=tg_nodes_c.std(0)
                            .unsqueeze(0)
                            .expand(sr_nodes_c.size()),
                        ).to(self.device)
                    )
                else:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(
                        self.device
                    )

                sr_nodes_c = self.seed_project_left(sr_nodes_c)
                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)
                sr_labels_category_first.append(
                    torch.ones(num_nodes, dtype=torch.float).to(self.device) * c
                )
                tg_labels_category_first.append(
                    torch.ones(num_nodes, dtype=torch.float).to(self.device) * c
                )
                sr_weight_category_first.append(
                    torch.ones(num_nodes, dtype=torch.long).to(self.device)
                )
                # tg_weight_category_first.append(tg_weight_c)

            elif (
                sr_indx.any()
            ):  # If there're no target nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(sr_nodes_c)

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_c = (
                    self.tg_seed[int(c.item())].unsqueeze(0).expand(num_nodes, 256)
                )

                if self.with_semantic_completion:
                    tg_nodes_c = (
                        torch.normal(0, 0.01, size=tg_nodes_c.size()).to(self.device)
                        + tg_nodes_c
                        if len(sr_nodes_c) < 5
                        else torch.normal(
                            mean=tg_nodes_c,
                            std=sr_nodes_c.std(0)
                            .unsqueeze(0)
                            .expand(sr_nodes_c.size()),
                        ).to(self.device)
                    )
                else:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).to(
                        self.device
                    )

                tg_nodes_c = self.seed_project_left(tg_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                sr_labels_category_first.append(
                    torch.ones(num_nodes, dtype=torch.float).to(self.device) * c
                )
                tg_labels_category_first.append(
                    torch.ones(num_nodes, dtype=torch.float).to(self.device) * c
                )

                # sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long))

        nodes_sr = torch.cat(sr_nodes_category_first, dim=0)
        nodes_tg = torch.cat(tg_nodes_category_first, dim=0)

        label_sr = torch.cat(sr_labels_category_first, dim=0)
        label_tg = torch.cat(tg_labels_category_first, dim=0)

        return nodes_sr, label_sr, nodes_tg, label_tg

    def _forward_intra_domain_graph(self, nodes):
        nodes, edges = self.intra_domain_graph(nodes, nodes, nodes)
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        nodes2_enahnced = self.cross_domain_graph(nodes_1, nodes_1, nodes_2)[0]
        nodes1_enahnced = self.cross_domain_graph(nodes_2, nodes_2, nodes_1)[0]

        return nodes1_enahnced, nodes2_enahnced

    def sinkhorn_rpm(self, log_alpha, n_iters=5, slack=True, eps=-1):
        """
        Run Sinkhorn iterations to generate a near doubly stochastic matrix.
        Args:
            log_alpha: Log of positive matrix to apply Sinkhorn normalization (B, J, K).
            n_iters (int): Number of normalization iterations.
            slack (bool): Whether to include slack row and column.
            eps: Epsilon for early termination (negative to disable).
        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K).
        """
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :]).squeeze(1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat(
                    (
                        log_alpha_padded[:, :-1, :]
                        - torch.logsumexp(
                            log_alpha_padded[:, :-1, :], dim=2, keepdim=True
                        ),
                        log_alpha_padded[:, -1, None, :],
                    ),
                    dim=1,
                )  # Don't normalize the last row

                # Column normalization
                log_alpha_padded = torch.cat(
                    (
                        log_alpha_padded[:, :, :-1]
                        - torch.logsumexp(
                            log_alpha_padded[:, :, :-1], dim=1, keepdim=True
                        ),
                        log_alpha_padded[:, :, -1, None],
                    ),
                    dim=2,
                )  # Don't normalize the last column

                if eps > 0 and prev_alpha is not None:
                    abs_dev = torch.abs(
                        torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha
                    )
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
                # Column normalization
                log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

                if eps > 0 and prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

        return log_alpha

    def _refine_graph_with_sinkhorn(self, nodes_1, labels_1, nodes_2, labels_2):
        """
        Refine the graph using self-attention and Sinkhorn optimization.
        """
        # Ensure nodes require gradients
        nodes_1 = nodes_1.clone().detach().requires_grad_()
        nodes_2 = nodes_2.clone().detach().requires_grad_()
        # Compute node affinity
        M = self.node_affinity(nodes_1, nodes_2)

        matching_target = torch.bmm(
            labels_1, labels_2.transpose(-1, -2)
        )  # (batch, num_samples * H * W, num_classes)

        matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()

        return matching_loss, M

    def _compute_node_loss(self, nodes, labels):
        """
        Compute the classification loss for nodes.
        """

        logits = self.node_cls(nodes)
        logits_expanded = logits.expand(
            -1, -1, labels.shape[2]
        )  # Shape: [batch_size, num_samples, channels_in]
        # Flatten both logits and ground truth to 1D
        logits_flat = logits_expanded.reshape(
            -1
        )  # Shape: [batch_size * num_samples * channels_in]
        ground_truth_flat = labels.reshape(
            -1
        ).float()  # Flatten and convert to float for BCE loss

        return self.node_loss_fn(logits_flat, ground_truth_flat.float())
