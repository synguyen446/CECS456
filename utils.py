import torch
from tqdm import tqdm

def fit(model, epochs,data_loader,val_loader,optimizer,loss_fn,scheduler=None, device='cpu'):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = val_running_loss =0.0
        correct = val_correct = 0
        total=val_total =0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, lables in progress_bar:
            images = images.to(device)
            lables = lables.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, lables)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs,1 )
            correct += (preds==lables).sum().item()
            total += lables.size(0)

            progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(correct/total):.4f}"
                    })

        epoch_loss = running_loss/total
        epoch_acc = correct/total

        model.eval()

        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, labels in val_bar:
                 images = images.to(device)
                 labels = labels.to(device)
        
                 outputs = model(images)
                 loss = loss_fn(outputs, labels)
        
                 val_running_loss += loss.item() * images.size(0)
                 _, preds = torch.max(outputs, 1)
                 val_correct += (preds == labels).sum().item()
                 val_total += labels.size(0)
        
                 val_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{(val_correct/val_total):.4f}"
                        })
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        if scheduler:
            scheduler.step(val_loss)
            
        if  best_val_loss > val_loss:
            best_val_loss = val_loss
            best_weight = model.state_dict()
            torch.save(best_weight,"best_update.pth")
            print(f"Saved new best model with val_loss = {best_val_loss:.4f}")

    return model

        
def eval(model, test_loader,loss_fn,device):
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
            
    val_bar = tqdm(test_loader, desc=f"Testing")
            
    with torch.no_grad():
        for images, labels in val_bar:
             images = images.to(device)
             labels = labels.to(device)
            
             outputs = model(images)
             loss = loss_fn(outputs, labels)
            
             val_running_loss += loss.item() * images.size(0)
             _, preds = torch.max(outputs, 1)
             val_correct += (preds == labels).sum().item()
             val_total += labels.size(0)
            
             val_bar.set_postfix({
                                "loss": f"{loss.item():.4f}",
                                "acc": f"{(val_correct/val_total):.4f}"
                            })
            
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
 
