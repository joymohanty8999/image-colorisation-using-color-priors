device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for hinted_imgs, target_imgs in train_loader:
            hinted_imgs, target_imgs = hinted_imgs.to(device), target_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(hinted_imgs)
            loss = criterion(outputs, target_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for hinted_imgs, target_imgs in valid_loader:
                hinted_imgs, target_imgs = hinted_imgs.to(device), target_imgs.to(device)
                outputs = model(hinted_imgs)
                loss = criterion(outputs, target_imgs)
                running_loss += loss.item()
        epoch_valid_loss = running_loss / len(valid_loader)  # Calculate average loss for the epoch
        valid_losses.append(epoch_valid_loss)

        scheduler.step(epoch_valid_loss)  # Ensure only the epoch's average loss is passed

        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}')

    return train_losses, valid_losses

train_losses, valid_losses = train_and_validate(
    model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs=10, device=device
)