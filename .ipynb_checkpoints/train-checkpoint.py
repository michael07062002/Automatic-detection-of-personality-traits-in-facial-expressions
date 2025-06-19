
import os, itertools, torch
from torch.utils.data import DataLoader
from emoformer.constants import DEVICE
from emoformer.datasets.fiv2_hemiface import FIV2HemifaceDS
from emoformer.models.emoformer import EmoFormer
from emoformer.losses.ccc_mse import LossCCC_MSE, concordance_cc


ROOT = "data/FirstImpressionsV2"

train_ds = FIV2HemifaceDS(
    root_videos = os.path.join(ROOT, "train"),
    anno_pkl    = os.path.join(ROOT, "annotation_training.pkl")
)
val_ds = FIV2HemifaceDS(
    root_videos = os.path.join(ROOT, "val"),
    anno_pkl    = os.path.join(ROOT, "annotation_validation.pkl")
)

tr_ld = DataLoader(train_ds, batch_size=4, shuffle=True,
                   num_workers=0, pin_memory=False)
val_ld = DataLoader(val_ds, batch_size=4, shuffle=False,
                    num_workers=0, pin_memory=False)


def train_epoch(model, loader, optimizer, loss_fn):
    model.train(); total = 0
    for segL, segR, y in loader:
        segL, segR, y = segL.to(DEVICE), segR.to(DEVICE), y.to(DEVICE)
        pred = model(segL, segR)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, loss_fn):
    model.eval(); total, preds, targets = 0, [], []
    with torch.no_grad():
        for segL, segR, y in loader:
            segL, segR, y = segL.to(DEVICE), segR.to(DEVICE), y.to(DEVICE)
            pred = model(segL, segR)
            loss = loss_fn(pred, y)
            total += loss.item()
            preds.append(pred); targets.append(y)
    preds = torch.cat(preds); targets = torch.cat(targets)
    ccc = concordance_cc(preds, targets).mean().item()
    return total / len(loader), ccc

param_grid = {
    'd_model': [256, 384, 512],
    'n_layer': [4, 6, 8],
    'alpha':   [0.1, 0.3],
    'lr':      [1e-4, 2e-4]
}
grid = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

best_ccc, best_params = -float('inf'), None

for i, values in enumerate(grid, 1):
    params = dict(zip(param_names, values))
    print(f"\nGrid {i}/{len(grid)}: {params}")

    model = EmoFormer(d_model=params['d_model'],
                      n_layer=params['n_layer']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=5)
    loss_fn = LossCCC_MSE(alpha=params['alpha'])

    best_val_ccc, epochs_no_improve = -float('inf'), 0

    for ep in range(1, 21):        
        tr_loss = train_epoch(model, tr_ld, optimizer, loss_fn)
        val_loss, val_ccc = eval_epoch(model, val_ld, loss_fn)

        print(f"  Ep{ep:02d}: train {tr_loss:.4f} | "
              f"val {val_loss:.4f} | CCC {val_ccc:.3f}")

        if val_ccc > best_val_ccc + 1e-4:
            best_val_ccc = val_ccc
            torch.save(model.state_dict(), "best_model_tmp.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(val_ccc)
        if epochs_no_improve >= 8:
            break

    print(f"Grid {i}: Best CCC = {best_val_ccc:.3f}")
    if best_val_ccc > best_ccc:
        best_ccc, best_params = best_val_ccc, params
        torch.save(model.state_dict(), "best_model_overall.pth")

print(f"\n Best CCC: {best_ccc:.3f}  with params: {best_params}")
