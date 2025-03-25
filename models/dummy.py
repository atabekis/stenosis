

class DummyTorchModel:
    mode = 'single_frame'

    def train_model(self, train_loader, val_loader):
        for epoch in range(2):
            print(f"Epoch {epoch+1} / 2")
            for batch_idx, (X, y, meta) in enumerate(train_loader):
                # X is shape: [batch_size, frames, 1, H, W]
                # y is shape: [batch_size, frames, 4]
                pass

    def evaluate(self, test_loader):
        for batch_idx, (X, y, meta) in enumerate(test_loader):
            pass
        return 0.0