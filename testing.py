def evaluate_on_test(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for hinted_imgs, target_imgs in test_loader:
            hinted_imgs, target_imgs = hinted_imgs.to(device), target_imgs.to(device)
            outputs = model(hinted_imgs)
            loss = criterion(outputs, target_imgs)
            running_loss += loss.item()
    test_loss = running_loss / len(test_loader)  # Calculate the average loss over all test batches
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss

test_loss = evaluate_on_test(model, test_loader, criterion, device)