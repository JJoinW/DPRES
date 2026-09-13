from transformers import BertModel, RobertaModel
import torch.nn as nn
import torch


class MultiLayer_Scorer(nn.Module):
    def __init__(self, args, model_path='bert', trait_num=4):
        super(MultiLayer_Scorer, self).__init__()
        self.args = args
        self.latent_dim = 768
        self.trait_num = trait_num

        self.dropout = nn.Dropout(p=self.args.dropout)

        if args.model_type == 'bert':
            self.encoder = BertModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                output_hidden_states=True,
                use_safetensors=True
            )
        else:
            self.encoder = RobertaModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                output_hidden_states=True,
                use_safetensors=True
            )

        self.trait_projection = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(trait_num)])
        self.scorers = nn.ModuleList(
            [nn.Linear(self.latent_dim + self.latent_dim, 1) for _ in range(trait_num)])
        self.shared_proj = nn.Linear(self.latent_dim, self.latent_dim)

        self.score_layer1 = nn.ModuleList(
            [nn.Linear(self.latent_dim, 1) for _ in range(trait_num - 1)])

        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.expert_gate = nn.Linear(self.latent_dim, self.trait_num)
        self.trait_experts = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.trait_num)])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        lhs = outputs.last_hidden_state
        lhs_cls = lhs[:, 0, :]

        shared_repr = self.shared_proj(lhs_cls)
        trait_projs = torch.stack([proj(lhs_cls) for proj in self.trait_projection], dim=1)
        trait_residuals = trait_projs - shared_repr.unsqueeze(1)

        trait_feas = trait_projs + trait_residuals
        expert_gate_outputs = torch.softmax(self.expert_gate(trait_feas), dim=1)  # [batch_size, expert_num]

        expert_selection = []
        for i in range(self.trait_num):
            moe_output = self.trait_experts[i](trait_feas)
            mask = torch.ones_like(moe_output)
            mask[:, i, :] = 0
            masked_output = moe_output * mask
            expert_selection.append(masked_output)

        expert_selection = torch.stack(expert_selection, dim=1).squeeze()

        if expert_gate_outputs.shape[0] == 1:
            expert_selection = expert_selection.unsqueeze(dim=0)

        trait_output = torch.einsum('btn,bntj->bnj', expert_gate_outputs, expert_selection)

        trait_reprs = torch.stack(
            [torch.cat((trait_feas[:, i, :], trait_output[:, i, :]), dim=-1) for i in range(self.trait_num)],
            dim=1).squeeze()

        if expert_gate_outputs.shape[0] == 1:
            trait_reprs = trait_reprs.unsqueeze(dim=0)

        scores = torch.stack(
            [self.sigmoid(self.scorers[i](self.dropout(trait_reprs[:, i, :]))) for i in range(self.trait_num)],
            dim=1,
        ).squeeze()

        return scores, trait_projs, shared_repr, trait_residuals, trait_reprs