from transformers import get_cosine_schedule_with_warmup
import torch.optim as optim
import torch

model = torch.nn.Linear(2, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.01)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps_per_epoch * 5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=200 * 50,
    num_cycles=1
)

for iter in range(200 * 50):
    optimizer.step()
    scheduler.step()
    if iter % 50 == 0:
        print(scheduler.get_last_lr()[-1])