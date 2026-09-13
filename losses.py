import torch
import torch.nn.functional as F


def correlation_coefficient(trait1, trait2):
    x = trait1
    y = trait2

    r_num = torch.sum(x * y)
    r_den = torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))

    r = torch.tensor(0.0)
    if r_den > 0:
        r = r_num / r_den
    return r


def SC_loss(pred_traits, scores):

    def trait_similarity(z_i, z_j, temperature=0.1):
        sim = F.cosine_similarity(z_i, z_j, dim=-1) / temperature
        return sim

    trait_num = scores.shape[-1]
    loss = 0.0

    for i in range(1, trait_num):
        h = pred_traits[:, i, :] 
        h_score = scores[:, i]           

        sim_scores = []

        for j in range(1, trait_num):
            if i != j:
                h_k = pred_traits[:, j, :]  
                h_k_score = scores[:, j]          

                corr_score = correlation_coefficient(h_score, h_k_score)
                sim_scores.append((h_k, corr_score))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        negative_h = sim_scores[-1][0]
        positive_hs = [item[0] for item in sim_scores[1:]]
        positive_cosine_sim = [torch.exp(trait_similarity(h.unsqueeze(0), h_k.unsqueeze(0))).sum() for h_k in positive_hs]
        positive_cosine_sim = torch.stack(positive_cosine_sim).sum()  
        negative_cosine_sim = torch.exp(trait_similarity(h.unsqueeze(0), negative_h.unsqueeze(0))).sum()

        consistency_term = positive_cosine_sim / (positive_cosine_sim + negative_cosine_sim)
        consistency_loss = -torch.log(consistency_term) / (trait_num - 2)

        loss += consistency_loss


    return loss / (trait_num - 1)  


def hidden_sim_loss(trait_hidden):

    def compute_similarity_matrix(traits_hidden):
        similarity_matrix = F.cosine_similarity(
            traits_hidden.unsqueeze(2),  # (batch_size, num_experts, 1, dimension)
            traits_hidden.unsqueeze(1),  # (batch_size, 1, num_experts, dimension)
            dim=-1
        )
        return similarity_matrix

    def select_positive_negative_pairs(similarity_matrix):
        batch_size, num_experts, _ = similarity_matrix.shape
        positive_pairs = []
        negative_pairs = []

        for b in range(batch_size):
            for i in range(num_experts):
 
                sim = similarity_matrix[b, i, :]
                sim[i] = -float('inf')  


                pos_j = torch.argmax(sim).item()
                positive_pairs.append((i, pos_j))

                neg_k = torch.argmin(sim).item()
                negative_pairs.append((i, neg_k))

        return positive_pairs, negative_pairs

    def trait_similarity(z_i, z_j, temperature=0.1):
        sim = F.cosine_similarity(z_i, z_j, dim=-1) / temperature
        return sim

    def loss_compute(expert_vectors, positive_pairs, negative_pairs, temperature=0.1):
        positive_sim_sum = 0
        negative_sim_sum = 0

        for (i, j) in positive_pairs:
            z_i = expert_vectors[:, i, :]  # 形状为 (batch_size, dimension)
            z_j = expert_vectors[:, j, :]  # 形状为 (batch_size, dimension)
            positive_sim_sum += torch.exp(trait_similarity(z_i, z_j, temperature)).sum()

        for (i, k) in negative_pairs:
            z_i = expert_vectors[:, i, :]  # 形状为 (batch_size, dimension)
            z_k = expert_vectors[:, k, :]  # 形状为 (batch_size, dimension)
            negative_sim_sum += torch.exp(trait_similarity(z_i, z_k, temperature)).sum()

        loss = -torch.log(positive_sim_sum / (positive_sim_sum + negative_sim_sum))
        return loss

    expert_similarity_matrix = compute_similarity_matrix(trait_hidden)
    batch_positive_pairs, batch_negative_pairs = select_positive_negative_pairs(expert_similarity_matrix)
    losses = loss_compute(trait_hidden, batch_positive_pairs, batch_negative_pairs)

    return losses


def mse_function(y_true, y_pred):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(y_true, y_pred)


def total_loss(y_pred, y_true):
    return mse_function(y_true, y_pred)


def orthogonality_loss(shared_repr, trait_residuals):
    # shared_repr: [B, D]
    # trait_residuals: [B, T, D]

    shared = F.normalize(shared_repr, dim=-1)
    residual = F.normalize(trait_residuals, dim=-1)

    shared = shared.unsqueeze(1)

    inner_product = torch.bmm(
        shared,
        residual.transpose(1,2)
    )

    loss = (inner_product ** 2).mean()

    return loss
