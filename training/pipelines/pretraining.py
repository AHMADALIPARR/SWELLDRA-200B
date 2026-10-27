import torch
def pretraining_loop(model, dataloader, optimizer, scheduler):
    model.train()
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].cuda()
        outputs = model(input_ids)
        lm_loss = torch.nn.functional.cross_entropy(outputs['logits'][:,:-1].contiguous().view(-1, outputs['logits'].shape[-1]), input_ids[:,1:].contiguous().view(-1), ignore_index=-100)
        loss = lm_loss + outputs['aux_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Step {step}: lm={lm_loss.item():.3f} aux={outputs['aux_loss'].item():.3f}")
