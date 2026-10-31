import torch
from torch import nn

from learn_pytorch import BlobDataset, MLPClassifier, TinyCNN, TrainConfig, fit, make_loaders
from learn_pytorch.export import export_program, load_checkpoint, load_program, save_checkpoint
from learn_pytorch.model import count_parameters


def test_dataset_item_and_batch_shapes():
    ds = BlobDataset(n_samples=100, n_classes=4, n_features=2, seed=1)
    x, y = ds[0]
    assert len(ds) == 100
    assert x.shape == (2,) and x.dtype == torch.float32
    assert y.shape == () and y.dtype == torch.int64

    train_loader, val_loader = make_loaders(ds, batch_size=16, val_fraction=0.2)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 2)
    assert yb.shape == (16,)
    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 20


def test_mlp_output_shape():
    model = MLPClassifier(in_features=2, num_classes=3)
    assert model(torch.randn(8, 2)).shape == (8, 3)


def test_cnn_output_shape():
    model = TinyCNN(num_classes=10)
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)
    assert count_parameters(model) > 0


def test_every_parameter_receives_a_gradient():
    model = MLPClassifier(in_features=2, num_classes=3)
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 2)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_short_training_run_learns():
    _, history = fit(TrainConfig(epochs=5, n_samples=600, device="cpu"))
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.val_acc[-1] > 0.9


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = MLPClassifier(2, 3).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)

    restored = load_checkpoint(MLPClassifier(2, 3), path)
    x = torch.randn(6, 2)
    torch.testing.assert_close(restored(x), model(x))


def test_export_roundtrip_with_dynamic_batch(tmp_path):
    torch.manual_seed(0)
    model = MLPClassifier(2, 3).eval()
    path = tmp_path / "model.pt2"
    export_program(model, torch.randn(4, 2), path)

    loaded = load_program(path)
    x = torch.randn(9, 2)  # a different batch size from the one used to export
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x), model(x))
