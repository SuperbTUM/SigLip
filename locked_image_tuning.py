import model
from constants import *
from data_preparation import *
from checkpoint import *
# from teacher import teacher_model_output
from losses import MMSupConAndProxyCE, SupConLoss, mine_hard_triplets
from eva02_clip import EVAReIDEncoder

import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List
import itertools

import os
import math
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptLearner(nn.Module):
    def __init__(self, 
                 text_tokenizer,
                 token_embedding,
                 num_prompt_tokens, 
                 embedding_dim, 
                 device,
                 class_names: List[str],
                 n_cls,
                 n_cams,
                 init_prompts: List[str] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.n_cls = n_cls
        self.n_cams = n_cams
        self.cam_embedding = nn.Embedding(n_cams, embedding_dim, device=device)
        nn.init.normal_(self.cam_embedding.weight, std=0.01)

        # Store tokenized class names
        self.class_names = class_names

        # --- Optional AI-generated prompts ---
        if init_prompts is not None:
            assert len(init_prompts) == len(class_names), \
                "init_prompts must match number of class names"
            self.init_prompts = init_prompts
        else:
            self.init_prompts = [f"a photo of a {name}" for name in class_names]
        # Note: In a full implementation, you'd handle tokenization carefully here.
        text_inputs = text_tokenizer(self.class_names, padding=True, return_tensors="pt").input_ids.to(device)
        ai_text_inputs = text_tokenizer(self.init_prompts, padding=True, return_tensors="pt").input_ids.to(device)

        self.register_buffer("text_inputs", text_inputs)
        self.register_buffer("ai_text_inputs", ai_text_inputs)

        # add positional embedding
        with torch.no_grad():
            class_embedding = token_embedding(text_inputs)
            ai_text_embedding = token_embedding(ai_text_inputs)[:, 1:, :]
        
        # Initialize the learnable prompt vectors
        init_prompt = torch.randn(n_cls, num_prompt_tokens, embedding_dim, dtype=ai_text_embedding.dtype, device=device)
        nn.init.normal_(init_prompt, std=0.02)
        self.prompt = nn.Parameter(init_prompt)

        self.register_buffer("class_name_embedding", class_embedding)
        self.register_buffer("ai_text_embedding", ai_text_embedding)

    def forward(self, text_model, label_pids, cam_ids):
        # --- Mix the AI-generated and learnable prompts ---
        # prompt: (N, P, D)
        # ai_prompt_embs: (N, L, D)
        # class_name_embs: (N, L', D)
        # Append the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        prompts = self.prompt[label_pids, :, :]
        cam_prompts = self.cam_embedding(cam_ids)
        prompts = F.dropout(prompts, p=0.05, training=self.training)
        class_name_embedding = self.class_name_embedding.expand(label_pids.size(0), -1, -1)
        combined_embs = torch.cat([prompts, class_name_embedding], dim=1)
        combined_embs[:, -1, :] = combined_embs[:, -1, :] + cam_prompts
        
        prompted_text_features = text_model(inputs_embeds=combined_embs)
        
        return prompted_text_features


def prompt_tuning_variable_dataset(base_model,
                                 dataset_names,
                                 input_sizes,
                                 class_names_list,
                                 device):
    # --- Shared model and optimizer ---
    if os.path.exists(f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth"):
        base_model, _, _, _, _, _ = load_checkpoint(base_model, None, None, None, None, f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth", device)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    frozen_text_model = copy.deepcopy(base_model.text_model.eval())
    
    vision_model = EVAReIDEncoder()
    vision_model.eval()
    base_model.text_model.train()
    for param in vision_model.parameters():
        param.requires_grad = False
    base_model = base_model.to(device)
    embedding_dim = base_model.config.text_config.hidden_size

    # --- Dataloaders and Prompt Learners for each dataset ---
    train_dataloaders = []
    prompt_learners = []
    full_prompts = []
    n_clses = []
    prompt_learner_params = []

    for dataset_name, input_size, class_names in zip(dataset_names, input_sizes, class_names_list):
        train_dataloader, _, n_cls, ai_prompts, n_cams = create_dataloader(dataset_name, input_size, "train", False, True)
        train_dataloaders.append(train_dataloader)
        n_clses.append(n_cls)
        full_prompts.append(text_tokenizer(ai_prompts, padding=True, return_tensors="pt").input_ids.to(device))

        if isinstance(class_names, str):
            class_names = [class_names]

        prompt_learner = PromptLearner(
            text_tokenizer=text_tokenizer,
            token_embedding=base_model.text_model.get_input_embeddings(),
            num_prompt_tokens=N_PROMPT_TOKEN,
            embedding_dim=embedding_dim,
            device=device,
            class_names=class_names,
            n_cls=n_cls,
            n_cams=n_cams
        )
        prompt_learner = prompt_learner.train()
        prompt_learners.append(prompt_learner)
        prompt_learner_params.extend(list(prompt_learner.parameters()))

    sup_con_loss = SupConLoss(device)
    scaler = GradScaler(device)
    prompt_learner_optimizer = torch.optim.Adam(
        prompt_learner_params + list(base_model.text_model.text_projection.parameters()), lr=3.5e-4, weight_decay=1e-4
    )
    temperature_optimizer = torch.optim.Adam(list(sup_con_loss.parameters()), lr=1e-4)

    prompt_scheduler = torch.optim.lr_scheduler.SequentialLR(
        prompt_learner_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                prompt_learner_optimizer,
                start_factor=0.1,      
                end_factor=1.0,
                total_iters=5
            ), 
            torch.optim.lr_scheduler.CosineAnnealingLR(
                prompt_learner_optimizer, 
                T_max=N_EPOCHS_LoRA-5, 
                eta_min=1e-6
            )
        ],
        milestones=[5]
    )
    
    # --- Pre-extract image features for all datasets ---
    image_features_lists = []
    image_label_lists = []
    image_cam_lists = []
    frozen_text_features_list = []
    with torch.inference_mode():
        batch_size = 32
        for n_cls, full_prompt in zip(n_clses, full_prompts):
            frozen_text_feature_list = []
            for i in range(0, len(full_prompt), batch_size):
                text_features = frozen_text_model(full_prompt[i:i+batch_size])
                frozen_text_feature_list.append(text_features)
            prompt_per_class = len(frozen_text_feature_list) // n_cls
            frozen_text_feature_list = torch.cat(frozen_text_feature_list, dim=0)
            if prompt_per_class > 1:
                frozen_text_feature_list = frozen_text_feature_list.view(len(full_prompt) // prompt_per_class, prompt_per_class, -1).mean(dim=1)
            frozen_text_features_list.append(frozen_text_feature_list)
        for i, train_dataloader in enumerate(train_dataloaders):
            image_features_list = []
            image_label_list = []
            image_cam_list = []
            for batch in train_dataloader:
                image_tensor, label, cam = batch[:3]
                image_tensor = image_tensor.to(device)
                label = label.to(device)
                cam = cam.to(device)
                image_features = vision_model(image_tensor)[0]
                image_features_list.append(image_features)
                image_label_list.append(label)
                image_cam_list.append(cam)
            image_features_lists.append(torch.cat(image_features_list, dim=0))
            image_label_lists.append(torch.cat(image_label_list, dim=0))
            image_cam_lists.append(torch.cat(image_cam_list, dim=0))
        # teacher_model_outputs = teacher_model_output(train_dataloaders, ai_prompts)
    
    # --- Training Loop ---
    samplers = [list(range(len(image_features_list))) for image_features_list in image_features_lists]
    for epoch in range(N_EPOCHS_LoRA):
        loss_by_epoch = 0

        for j in range(len(samplers)):
            random.shuffle(samplers[j])
        dataset_indices = [0 for _ in range(len(dataset_names))]
        num_batches = max(math.ceil(len(sampler) / BATCH_SIZE) for sampler in samplers) * len(samplers)
        
        for i in range(num_batches):
            dataset_idx = i % len(samplers)
            interval_start = dataset_indices[dataset_idx]
            if interval_start == len(image_cam_lists[dataset_idx]):
                continue
            interval_end = min(interval_start+BATCH_SIZE, len(image_cam_lists[dataset_idx]))
            batched_indices = samplers[dataset_idx][interval_start:interval_end]

            prompt_learner_optimizer.zero_grad()
            temperature_optimizer.zero_grad()
            total_loss = 0
            
            image_features_batch = image_features_lists[dataset_idx][batched_indices]
            image_cams_batch = image_cam_lists[dataset_idx][batched_indices]
            label_batch = image_label_lists[dataset_idx][batched_indices]
            dataset_indices[dataset_idx] = interval_end
            # frozen_text_features_batch = frozen_text_features_list[dataset_idx][label_batch]

            with autocast(device):
                text_features = prompt_learners[dataset_idx](base_model.text_model, label_batch, image_cams_batch)
            
            text_features = text_features.float()
            label = label_batch 
            loss = sup_con_loss(image_features_batch, text_features, label, label) + \
            sup_con_loss(text_features, image_features_batch, label, label)
            total_loss += loss

            scaler.scale(total_loss).backward()
            scaler.step(prompt_learner_optimizer)
            scaler.step(temperature_optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
        prompt_scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    
    for prompt_learner in prompt_learners:
        prompt_learner.eval()
    save_checkpoint(base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item(), N_EPOCHS_LoRA, None, prompt_learner_optimizer, prompt_scheduler)
    return base_model.eval(), prompt_learners

