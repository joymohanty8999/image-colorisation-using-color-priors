batch_size = 32
num_workers = 4
shuffle_train = True
shuffle_valid = False
shuffle_test = False

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle_train,
    num_workers=num_workers,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=shuffle_valid,
    num_workers=num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=shuffle_test,
    num_workers=num_workers,
    pin_memory=True
)