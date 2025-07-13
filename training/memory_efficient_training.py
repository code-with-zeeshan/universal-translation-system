# training/memory_efficient_training.py

def optimize_memory_usage():
    """Techniques to reduce memory usage"""
    
    # 1. Gradient Checkpointing
    model.gradient_checkpointing_enable()
    
    # 2. Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()
    
    # 3. Gradient Accumulation
    accumulation_steps = 8  # Simulate larger batch
    
    # 4. CPU Offloading for optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
        fused=True  # Faster on A100
    )
    
    # 5. Dynamic Batch Size
    batch_size = get_dynamic_batch_size(
        available_memory=torch.cuda.get_device_properties(0).total_memory,
        model_size=sum(p.numel() for p in model.parameters())
    )
    
    # 6. Efficient Data Loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True  # Shuffle data for dynamic batching
    )
    
    # 7. Model Parallelism
    model = nn.DataParallel(model)
