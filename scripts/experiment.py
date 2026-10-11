# Exported from the clean notebook. Edit paths and flags before running.

import os, re, json, hashlib, platform, subprocess, sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from IPython.display import display
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, '| PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)


USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

DATA_DIR = Path('/content/data/AD Datasets/Data')
METADATA_PATH = Path('/content/data/oasis_cross-sectional.xlsx')
RAD_WEIGHTS = Path('/content/weights/RadImageNet-ResNet50_notop_torch.pth')
OUTPUT_DIR = Path('/content/transfer_learning_runs/run_01')
RUN_TRAINING = False
RUN_INFERENCE = False
RUN_BOOTSTRAP = False

IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 50
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 10
SEED = 42
N_FOLDS = 5
N_BOOT = 10_000
NUM_CLASSES = 3
CLASS_MAP = {'Non Demented': 0, 'Very mild Dementia': 1, 'Mild Dementia': 2, 'Moderate Dementia': 2}
CLASS_NAMES = ['Non-Demented', 'Very Mild Dementia', 'Mild+Moderate Dementia']
CONDITIONS = [('scratch','full'), ('imagenet','frozen'), ('imagenet','partial'), ('imagenet','full'),
              ('radimagenet','frozen'), ('radimagenet','partial'), ('radimagenet','full')]
OOF_CONDITIONS = [('scratch','full'), ('imagenet','frozen'), ('radimagenet','frozen')]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
if RUN_TRAINING and DEVICE.type != 'cuda':
    raise RuntimeError('Select a GPU runtime before enabling training.')


PATTERN = re.compile(r'^(OAS1_\d{4})_(MR\d+)_(mpr-\d+)_(\d+)\.(jpg|jpeg|png|bmp)$', re.IGNORECASE)
manifest_path = OUTPUT_DIR / 'manifest.csv'
if manifest_path.exists():
    manifest = pd.read_csv(manifest_path)
else:
    records, unmatched = [], []
    for folder, label in CLASS_MAP.items():
        directory = DATA_DIR / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for filename in os.listdir(directory):
            match = PATTERN.match(filename)
            if match is None:
                if Path(filename).suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
                    unmatched.append(filename)
                continue
            records.append(dict(filepath=str(directory / filename), filename=filename, class_folder=folder,
                label=label, subject_id=match[1], session=match[2], acquisition=match[3], slice_num=int(match[4])))
    if unmatched:
        raise ValueError(f'Unrecognized slice names (first 5): {unmatched[:5]}')
    manifest = pd.DataFrame(records)
assert len(manifest) == 86437, 'The paper cohort contains 86,437 slices.'
assert manifest.groupby('subject_id').label.nunique().eq(1).all()
subjects = manifest[['subject_id','label']].drop_duplicates().sort_values('subject_id').reset_index(drop=True)
assert subjects.label.value_counts().sort_index().to_dict() == {0:266,1:58,2:23}
assert manifest.filepath.map(lambda x: Path(x).is_file()).all(), 'A saved image path is missing.'
manifest.to_csv(manifest_path, index=False)
display(subjects.groupby('label').size().rename('subjects').to_frame())
print('Sessions:', len(manifest[['subject_id','session']].drop_duplicates()))
print('Acquisitions:', len(manifest[['subject_id','session','acquisition']].drop_duplicates()))


def make_subject_splits(subject_table):
    subject_table = subject_table.sort_values('subject_id').reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for trainval_idx, test_idx in outer.split(np.arange(len(subject_table)), subject_table.label, subject_table.subject_id):
        trainval = subject_table.iloc[trainval_idx]
        inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
        tr, va = next(inner.split(np.arange(len(trainval)), trainval.label, trainval.subject_id))
        fold = {'train':trainval.iloc[tr].subject_id.tolist(), 'val':trainval.iloc[va].subject_id.tolist(),
                'test':subject_table.iloc[test_idx].subject_id.tolist()}
        assert not set(fold['train']) & set(fold['val'])
        assert not set(fold['train']) & set(fold['test'])
        assert not set(fold['val']) & set(fold['test'])
        splits.append(fold)
    held_out = [sid for fold in splits for sid in fold['test']]
    assert len(held_out) == len(set(held_out)) == len(subject_table)
    return splits

full_splits = make_subject_splits(subjects)
(OUTPUT_DIR / 'fold_splits.json').write_text(json.dumps(full_splits, indent=2))
print([(len(f['train']),len(f['val']),len(f['test'])) for f in full_splits])


# ImageNet normalization (used for all conditions including scratch — standard practice)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class OASISDataset(Dataset):
    """Returns (image, label, subject_id) — subject_id needed for aggregation."""

    def __init__(self, manifest_df, subject_ids, transform=None):
        self.df = manifest_df[
            manifest_df['subject_id'].isin(subject_ids)
        ].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label']), str(row['subject_id'])

def make_loaders(fold_dict, manifest_df, label_frac=1.0, seed=SEED):
    """Build train/val/test DataLoaders for one fold.
    label_frac: fraction of TRAINING SUBJECTS to use (for data-efficiency ablations).
    """
    rng = np.random.default_rng(seed)

    train_subjs = np.array(fold_dict['train'])
    val_subjs   = np.array(fold_dict['val'])
    test_subjs  = np.array(fold_dict['test'])

    # Stratified subject subsample for label fraction ablations
    if label_frac < 1.0:
        st = manifest_df[manifest_df['subject_id'].isin(train_subjs)] \
             .drop_duplicates('subject_id')[['subject_id','label']]
        keep = []
        for lbl in st['label'].unique():
            s = st[st['label']==lbl]['subject_id'].values
            k = max(1, int(len(s) * label_frac))
            keep.extend(rng.choice(s, k, replace=False).tolist())
        train_subjs = np.array(keep)

    train_ds = OASISDataset(manifest_df, train_subjs, train_transform)
    val_ds   = OASISDataset(manifest_df, val_subjs,   val_transform)
    test_ds  = OASISDataset(manifest_df, test_subjs,  val_transform)

    # Class-balanced loss weights from training labels
    train_labels = train_ds.df['label'].values
    cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, class_weights


def load_radimagenet_weights(model, path):
    """Load RadImageNet-ResNet50_notop weights into a torchvision ResNet-50."""
    raw = torch.load(path, map_location='cpu')

    stage_map = {
        'backbone.0.': 'conv1.',
        'backbone.1.': 'bn1.',
        'backbone.4.': 'layer1.',
        'backbone.5.': 'layer2.',
        'backbone.6.': 'layer3.',
        'backbone.7.': 'layer4.',
    }

    remapped = OrderedDict()
    for k, v in raw.items():
        new_k = None
        for old, new in stage_map.items():
            if k.startswith(old):
                new_k = new + k[len(old):]
                break
        if new_k is not None:
            remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    backbone_keys = [k for k in model.state_dict() if not k.startswith('fc.')]
    coverage = (len(backbone_keys) - len([k for k in missing if not k.startswith('fc.')])) / len(backbone_keys)

    assert coverage >= 0.99, f"RadImageNet backbone coverage {coverage:.1%} — abort"
    print(f"  RadImageNet loaded: backbone coverage {coverage:.1%}, missing={missing}, unexpected={unexpected}")
    return model

def build_model(init_strategy, fine_tune='full', num_classes=NUM_CLASSES):
    """
    Build ResNet-50 with specified initialization and fine-tuning strategy.

    init_strategy : 'scratch' | 'imagenet' | 'radimagenet'
    fine_tune     : 'full' | 'partial' | 'frozen'
                    NOTE: frozen/partial are forced to 'full' for scratch
                    (untrained frozen weights produce meaningless features)
    """
    if init_strategy == 'scratch':
        model = models.resnet50(weights=None)
        if fine_tune != 'full':
            print(f'  [scratch] fine_tune="{fine_tune}" forced to "full" (frozen scratch = meaningless)')
            fine_tune = 'full'

    elif init_strategy == 'imagenet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    elif init_strategy == 'radimagenet':
        model = models.resnet50(weights=None)
        if not os.path.exists(RAD_WEIGHTS):
            raise FileNotFoundError(
                f'RadImageNet weights not found:\n  {RAD_WEIGHTS}\n'
                'Download from https://github.com/BMEII-AI/RadImageNet\n'
                'File: RadImageNet-ResNet50_notop_torch.pth'
            )
        model = load_radimagenet_weights(model, RAD_WEIGHTS)
    else:
        raise ValueError(f'Unknown init_strategy: {init_strategy}')

    if fine_tune == 'frozen':
        for p in model.parameters():
            p.requires_grad = False
    elif fine_tune == 'partial':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer3.parameters():   # ResNet50 blocks 3+4
            p.requires_grad = True
        for p in model.layer4.parameters():
            p.requires_grad = True
    # 'full' → all parameters trainable (default ResNet state)

    in_features = model.fc.in_features   # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    for p in model.fc.parameters():
        p.requires_grad = True

    model = model.to(DEVICE)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  [{init_strategy}/{fine_tune}] trainable params: {n_train:,} / {n_total:,}')

    return model, fine_tune

def build_optimizer(model, fine_tune):
    head_params     = list(model.fc.parameters())
    backbone_params = [p for n,p in model.named_parameters()
                       if 'fc' not in n and p.requires_grad]
    return torch.optim.AdamW(
        [{'params': head_params,     'lr': LR_HEAD},
         {'params': backbone_params, 'lr': LR_BACKBONE}],
        weight_decay=WEIGHT_DECAY
    )


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    Subject-level evaluation:
      1. Collect softmax prob vectors for ALL slices
      2. For each subject, take the MEAN probability vector (not majority vote)
         — mean preserves confidence information; enables proper AUC computation
      3. Compute subject-level F1, AUC, Accuracy
    """
    model.eval()
    all_probs, all_labels, all_subjs = [], [], []
    total_loss, n = 0.0, 0

    for images, labels, subj_ids in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1).cpu().numpy()

        total_loss += loss.item() * len(labels)
        n          += len(labels)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_subjs.extend(list(subj_ids))

    all_probs  = np.array(all_probs)    # (N_slices, C)
    all_labels = np.array(all_labels)   # (N_slices,)
    all_subjs  = np.array(all_subjs)    # (N_slices,)

    unique_subjs = np.unique(all_subjs)
    subj_probs, subj_labels = [], []
    for sid in unique_subjs:
        mask = (all_subjs == sid)
        subj_probs.append(all_probs[mask].mean(axis=0))   # mean over slices
        subj_labels.append(int(all_labels[mask][0]))

    subj_probs  = np.array(subj_probs)  # (S, C)
    subj_preds  = subj_probs.argmax(axis=1)
    subj_labels = np.array(subj_labels)

    subj_f1  = f1_score(subj_labels, subj_preds, average='macro', zero_division=0)
    subj_acc = (subj_preds == subj_labels).mean()

    try:
        subj_auc = roc_auc_score(
            subj_labels, subj_probs,
            multi_class='ovr', average='macro'
        )
    except ValueError:
        subj_auc = float('nan')

    # Slice-level accuracy is logged only; subject metrics are the main outcomes.
    slice_acc = (all_probs.argmax(axis=1) == all_labels).mean()

    return {
        'slice_loss':   total_loss / n,
        'slice_acc':    slice_acc,
        'subj_f1':      subj_f1,
        'subj_acc':     subj_acc,
        'subj_auc':     subj_auc,
        'subj_probs':   subj_probs,
        'subj_preds':   subj_preds,
        'subj_labels':  subj_labels,
    }


def run_condition(init_strategy, fine_tune, manifest_df, splits, output_dir, label_frac=1.0):

    full_csv = f'{output_dir}/{init_strategy}_{fine_tune}_frac{label_frac}_results.csv'
    if os.path.exists(full_csv):
        print(f'  ✓ Condition {init_strategy}/{fine_tune} already complete — loading from cache')
        return pd.read_csv(full_csv)

    fold_results = []
    effective_ft = fine_tune  # updated per fold if build_model overrides it

    for fold_i, fold_dict in enumerate(splits):
        fold_csv  = (f'{output_dir}/{init_strategy}_{fine_tune}'
                     f'_frac{label_frac}_fold{fold_i+1}_result.csv')
        ckpt_path = (f'{output_dir}/{init_strategy}_{fine_tune}'
                     f'_frac{label_frac}_fold{fold_i+1}.pth')

        if os.path.exists(fold_csv):
            print(f'\n── Fold {fold_i+1}/{N_FOLDS} — result cached, skipping')
            fold_results.append(pd.read_csv(fold_csv).to_dict('records')[0])
            continue

        #    Re-evaluate test set from saved weights (~1 min, no retraining)
        if os.path.exists(ckpt_path):
            print(f'\n── Fold {fold_i+1}/{N_FOLDS} — checkpoint found, re-evaluating …')
            train_ldr, val_ldr, test_ldr, class_wts = make_loaders(
                fold_dict, manifest_df, label_frac=label_frac, seed=SEED + fold_i)
            model, eff_ft = build_model(init_strategy, fine_tune)
            effective_ft  = eff_ft
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            criterion = nn.CrossEntropyLoss(weight=class_wts)
            test_m = evaluate(model, test_ldr, criterion)
            print(f'   TEST → subj_F1={test_m["subj_f1"]:.4f} '
                  f'subj_AUC={test_m["subj_auc"]:.4f} '
                  f'subj_Acc={test_m["subj_acc"]:.4f} '
                  f'[slice_Acc={test_m["slice_acc"]:.4f}]')
            row = {
                'init':             init_strategy,
                'fine_tune':        eff_ft,
                'label_frac':       label_frac,
                'fold':             fold_i + 1,
                'best_val_subj_f1': float('nan'),
                'test_subj_f1':     test_m['subj_f1'],
                'test_subj_auc':    test_m['subj_auc'],
                'test_subj_acc':    test_m['subj_acc'],
                'test_slice_acc':   test_m['slice_acc'],
            }
            fold_results.append(row)
            pd.DataFrame([row]).to_csv(fold_csv, index=False)
            del model
            torch.cuda.empty_cache()
            continue

        print(f'\n── Fold {fold_i+1}/{N_FOLDS} ─────────────────────────────────────')
        train_ldr, val_ldr, test_ldr, class_wts = make_loaders(
            fold_dict, manifest_df, label_frac=label_frac, seed=SEED + fold_i)
        print(f'   train slices: {len(train_ldr.dataset):,} | '
              f'val: {len(val_ldr.dataset):,} | test: {len(test_ldr.dataset):,}')

        model, effective_ft = build_model(init_strategy, fine_tune)
        criterion  = nn.CrossEntropyLoss(weight=class_wts)
        optimizer  = build_optimizer(model, effective_ft)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5)

        best_val_f1    = -1.0
        best_state     = None
        patience_count = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, train_ldr, optimizer, criterion)
            val_m           = evaluate(model, val_ldr, criterion)
            val_f1          = val_m['subj_f1']
            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1    = val_f1
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f'   E{epoch:02d}: loss={tr_loss:.4f} tr_acc={tr_acc:.3f} '
                      f'val_F1={val_f1:.3f} val_AUC={val_m["subj_auc"]:.3f} '
                      f'best={best_val_f1:.3f} patience={patience_count}')

            if patience_count >= PATIENCE:
                print(f'   → Early stop at epoch {epoch} (best val_F1={best_val_f1:.4f})')
                break

        model.load_state_dict(best_state)
        test_m = evaluate(model, test_ldr, criterion)

        print(f'   TEST → subj_F1={test_m["subj_f1"]:.4f} '
              f'subj_AUC={test_m["subj_auc"]:.4f} '
              f'subj_Acc={test_m["subj_acc"]:.4f} '
              f'[slice_Acc={test_m["slice_acc"]:.4f}]')

        row = {
            'init':             init_strategy,
            'fine_tune':        effective_ft,
            'label_frac':       label_frac,
            'fold':             fold_i + 1,
            'best_val_subj_f1': best_val_f1,
            'test_subj_f1':     test_m['subj_f1'],
            'test_subj_auc':    test_m['subj_auc'],
            'test_subj_acc':    test_m['subj_acc'],
            'test_slice_acc':   test_m['slice_acc'],
        }
        fold_results.append(row)

        # Save per-fold result CSV (enables fold-level resume next time)
        pd.DataFrame([row]).to_csv(fold_csv, index=False)
        # Save checkpoint
        torch.save(best_state, ckpt_path)

        del model
        torch.cuda.empty_cache()

    result_df = pd.DataFrame(fold_results)
    result_df.to_csv(full_csv, index=False)
    return result_df


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

config = dict(seed=SEED, batch_size=BATCH_SIZE, epochs=EPOCHS, lr_head=LR_HEAD,
    lr_backbone=LR_BACKBONE, weight_decay=WEIGHT_DECAY, patience=PATIENCE, num_workers=NUM_WORKERS,
    torch=torch.__version__, torchvision=torchvision.__version__,
    manifest_sha256=sha256_file(manifest_path), splits=full_splits,
    metadata_sha256=sha256_file(METADATA_PATH), radimagenet_sha256=sha256_file(RAD_WEIGHTS),
    implementation='cleaned-notebook-v1')
config_path = OUTPUT_DIR / 'run_config.json'
if config_path.exists() and json.loads(config_path.read_text()) != config:
    raise RuntimeError('Configuration changed. Use a new OUTPUT_DIR; do not delete old results.')
config_path.write_text(json.dumps(config, indent=2))
(OUTPUT_DIR / 'environment.json').write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
    torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), indent=2))
(OUTPUT_DIR / 'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))


def summarize_folds(frame):
    counts = frame.groupby(['init','fine_tune','label_frac']).fold.nunique()
    assert counts.eq(5).all(), 'Each configuration needs five completed folds.'
    assert not frame.duplicated(['init','fine_tune','label_frac','fold']).any()
    return frame.groupby(['init','fine_tune','label_frac'])[['test_subj_f1','test_subj_auc','test_subj_acc']].agg(['mean','std'])

if RUN_TRAINING:
    full_results = pd.concat([run_condition(init,ft,manifest,full_splits,OUTPUT_DIR)
                              for init,ft in CONDITIONS], ignore_index=True)
    full_results.to_csv(OUTPUT_DIR / 'full_results.csv', index=False)
    display(summarize_folds(full_results))
else:
    print('Full-cohort training is off.')


if RUN_TRAINING:
    fraction_results = pd.concat([run_condition(init,ft,manifest,full_splits,OUTPUT_DIR,label_frac=frac)
        for init,ft in OOF_CONDITIONS for frac in [0.25,0.5,1.0]], ignore_index=True)
    fraction_results.to_csv(OUTPUT_DIR / 'ablation_results.csv', index=False)
    display(summarize_folds(fraction_results))
    paired = fraction_results.pivot(index=['label_frac','fold'],columns='init',values='test_subj_f1')
    print('Folds where frozen RadImageNet exceeds frozen ImageNet:')
    print((paired.radimagenet > paired.imagenet).groupby('label_frac').sum())
else:
    print('Label-fraction training is off.')


meta = pd.read_excel(METADATA_PATH) if METADATA_PATH.suffix.lower() in {'.xls','.xlsx'} else pd.read_csv(METADATA_PATH)
required = {'ID','Age','M/F','CDR','MMSE','Educ'}
assert required <= set(meta.columns), f'Missing metadata columns: {required-set(meta.columns)}'
meta['subject_id'] = meta.ID.astype(str).str.extract(r'(OAS1_\d{4})', expand=False).str.upper()
meta['_MR_NUMBER'] = pd.to_numeric(meta.ID.astype(str).str.extract(r'_MR(\d+)',expand=False), errors='coerce')
meta_subject = meta.sort_values(['subject_id','_MR_NUMBER']).drop_duplicates('subject_id')
audit = subjects.rename(columns={'label':'true_label'}).merge(meta_subject,on='subject_id',how='left',validate='one_to_one')
assert audit.ID.notna().all(), 'Some analysis subjects do not match the metadata.'
for col in ['Age','CDR','MMSE','Educ']:
    audit[col] = pd.to_numeric(audit[col],errors='coerce')
assert audit.Age.notna().all()
display(audit.groupby('true_label').Age.agg(['count','mean','std']))
display(audit[['Age','M/F','CDR','MMSE','Educ']].notna().sum().rename('available').to_frame())
known = audit[audit.CDR.notna()]
match = ((known.true_label==0)&(known.CDR==0)) | ((known.true_label==1)&(known.CDR==0.5)) | ((known.true_label==2)&known.CDR.isin([1,2]))
print('CDR agreement:', int(match.sum()), '/', len(known))
print('Non-Demented subjects younger than 60:', int(((audit.true_label==0)&(audit.Age<60)).sum()))
audit.to_csv(OUTPUT_DIR / 'demographic_audit.csv',index=False)


age_subjects = audit.loc[audit.Age>=60,['subject_id','true_label']].rename(columns={'true_label':'label'}).sort_values('subject_id').reset_index(drop=True)
assert age_subjects.label.value_counts().sort_index().to_dict()=={0:85,1:58,2:23}
age_manifest = manifest[manifest.subject_id.isin(age_subjects.subject_id)].copy()
age_splits = make_subject_splits(age_subjects)
AGE_OUTPUT_DIR = OUTPUT_DIR / 'age60_sensitivity'
AGE_OUTPUT_DIR.mkdir(exist_ok=True)
(AGE_OUTPUT_DIR / 'fold_splits.json').write_text(json.dumps(age_splits,indent=2))
if RUN_TRAINING:
    age_results = pd.concat([run_condition(init,ft,age_manifest,age_splits,AGE_OUTPUT_DIR)
                            for init,ft in CONDITIONS],ignore_index=True)
    age_results.to_csv(AGE_OUTPUT_DIR / 'full_results.csv',index=False)
    display(summarize_folds(age_results))
else:
    print('Age-restricted training is off.')


DEVICE_OOF = DEVICE
OOF_BATCH_SIZE = 64
def build_model_for_checkpoint(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model.to(DEVICE_OOF)

@torch.inference_mode()
def predict_subjects_oof(model, loader):

    model.eval()

    all_probs = []
    all_labels = []
    all_subjects = []

    for batch_i, (images, labels, subject_ids) in enumerate(loader):

        images = images.to(DEVICE_OOF)

        logits = model(images)

        probs = F.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.extend(labels.numpy().tolist())
        all_subjects.extend(list(subject_ids))

        if (batch_i + 1) % 50 == 0:
            print(
                f"      processed "
                f"{min((batch_i+1)*OOF_BATCH_SIZE, len(loader.dataset)):,}"
                f"/{len(loader.dataset):,} slices"
            )

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.asarray(all_labels)
    all_subjects = np.asarray(all_subjects)

    rows = []

    for sid in np.unique(all_subjects):

        mask = (all_subjects == sid)

        mean_prob = all_probs[mask].mean(axis=0)
        true_label = int(all_labels[mask][0])
        pred_label = int(np.argmax(mean_prob))

        row = {
            "subject_id": sid,
            "true_label": true_label,
            "pred_label": pred_label,
            "n_slices": int(mask.sum())
        }

        for c in range(NUM_CLASSES):
            row[f"prob_class_{c}"] = float(mean_prob[c])

        rows.append(row)

    return pd.DataFrame(rows)

def generate_oof(init,ft,manifest_df,subject_table,splits,output_dir):
    path = output_dir / f'OOF_{init}_{ft}_subject_predictions.csv'
    if path.exists():
        result = pd.read_csv(path)
    else:
        rows = []
        for fold_i,fold in enumerate(splits,1):
            checkpoint = output_dir / f'{init}_{ft}_frac1.0_fold{fold_i}.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Complete training or provide the matching checkpoint: {checkpoint}')
            model = build_model_for_checkpoint()
            model.load_state_dict(torch.load(checkpoint,map_location=DEVICE_OOF),strict=True)
            loader = DataLoader(OASISDataset(manifest_df,fold['test'],val_transform),
                                batch_size=OOF_BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
            pred = predict_subjects_oof(model,loader)
            pred['fold'],pred['init'],pred['fine_tune'] = fold_i,init,ft
            rows.append(pred)
            del model
        result = pd.concat(rows,ignore_index=True)
    assert not result.subject_id.duplicated().any()
    assert set(result.subject_id)==set(subject_table.subject_id)
    labels = subject_table.set_index('subject_id').label
    assert np.array_equal(result.true_label.to_numpy(),labels.loc[result.subject_id].to_numpy())
    expected_fold = {sid:i for i,f in enumerate(splits,1) for sid in f['test']}
    assert result.fold.eq(result.subject_id.map(expected_fold)).all()
    probs = result[[f'prob_class_{c}' for c in range(3)]].to_numpy()
    assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(axis=1),1,atol=1e-5)
    assert np.array_equal(result.pred_label.to_numpy(),probs.argmax(axis=1))
    result.to_csv(path,index=False)
    return result

cohort_inputs = {'full':(manifest,subjects,full_splits,OUTPUT_DIR),
                 'age60':(age_manifest,age_subjects,age_splits,AGE_OUTPUT_DIR)}
if RUN_INFERENCE:
    for cohort_name,args in cohort_inputs.items():
        for init,ft in OOF_CONDITIONS:
            generate_oof(init,ft,*args)
else:
    print('OOF inference is off.')


def calculate_metric(
    y_true,
    y_pred,
    y_prob,
    metric
):

    if metric == "macro_f1":

        return f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

    elif metric == "macro_auc":

        return roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    elif metric == "balanced_accuracy":

        return balanced_accuracy_score(
            y_true,
            y_pred
        )

    else:

        raise ValueError(
            f"Unknown metric: {metric}"
        )

def align_oof_predictions(
    model_a,
    model_b
):

    assert not model_a["subject_id"].duplicated().any()
    assert not model_b["subject_id"].duplicated().any()
    a = (
        model_a
        .sort_values("subject_id")
        .reset_index(drop=True)
        .copy()
    )

    b = (
        model_b
        .sort_values("subject_id")
        .reset_index(drop=True)
        .copy()
    )

    # Same subjects
    assert np.array_equal(
        a["subject_id"].to_numpy(),
        b["subject_id"].to_numpy()
    ), "Subject IDs do not align between models."

    # Same true labels
    assert np.array_equal(
        a["true_label"].to_numpy(),
        b["true_label"].to_numpy()
    ), "True labels differ between models."

    # Same held-out fold
    if "fold" in a.columns and "fold" in b.columns:

        assert np.array_equal(
            a["fold"].to_numpy(),
            b["fold"].to_numpy()
        ), "Fold assignments differ between models."

    return a, b

def paired_stratified_bootstrap(
    model_a,
    model_b,
    metric="macro_f1",
    n_boot=10000,
    seed=42
):

    # --------------------------------------------
    # Align subjects
    # --------------------------------------------
    a, b = align_oof_predictions(
        model_a,
        model_b
    )

    y_true = (
        a["true_label"]
        .to_numpy()
        .astype(int)
    )

    pred_a = (
        a["pred_label"]
        .to_numpy()
        .astype(int)
    )

    pred_b = (
        b["pred_label"]
        .to_numpy()
        .astype(int)
    )

    probability_columns = [
        f"prob_class_{c}"
        for c in range(NUM_CLASSES)
    ]

    prob_a = (
        a[probability_columns]
        .to_numpy()
        .astype(float)
    )

    prob_b = (
        b[probability_columns]
        .to_numpy()
        .astype(float)
    )

    # --------------------------------------------
    # Observed metrics
    # --------------------------------------------
    observed_a = calculate_metric(
        y_true,
        pred_a,
        prob_a,
        metric
    )

    observed_b = calculate_metric(
        y_true,
        pred_b,
        prob_b,
        metric
    )

    observed_difference = (
        observed_a - observed_b
    )

    # --------------------------------------------
    # Subject indices for each true class
    #
    # Stratified bootstrap preserves the original
    # number of subjects in each diagnostic class.
    # --------------------------------------------
    unique_classes = np.unique(y_true)

    class_indices = {

        c: np.where(
            y_true == c
        )[0]

        for c in unique_classes
    }

    # Verify all 3 classes are present
    assert len(class_indices) == NUM_CLASSES

    # --------------------------------------------
    # Bootstrap
    # --------------------------------------------
    rng = np.random.default_rng(seed)

    bootstrap_differences = np.empty(
        n_boot,
        dtype=float
    )

    for boot_i in range(n_boot):

        sampled_indices = []

        # Sample WITH replacement within each class
        for c in unique_classes:

            idx = class_indices[c]

            sampled_class_indices = rng.choice(
                idx,
                size=len(idx),
                replace=True
            )

            sampled_indices.append(
                sampled_class_indices
            )

        sampled_indices = np.concatenate(
            sampled_indices
        )

        # ----------------------------------------
        # SAME resampled subjects are used
        # for both models -> paired bootstrap
        # ----------------------------------------
        boot_y_true = (
            y_true[sampled_indices]
        )

        boot_pred_a = (
            pred_a[sampled_indices]
        )

        boot_pred_b = (
            pred_b[sampled_indices]
        )

        boot_prob_a = (
            prob_a[sampled_indices]
        )

        boot_prob_b = (
            prob_b[sampled_indices]
        )

        # ----------------------------------------
        # Calculate bootstrap metrics
        # ----------------------------------------
        metric_a = calculate_metric(
            boot_y_true,
            boot_pred_a,
            boot_prob_a,
            metric
        )

        metric_b = calculate_metric(
            boot_y_true,
            boot_pred_b,
            boot_prob_b,
            metric
        )

        bootstrap_differences[boot_i] = (
            metric_a - metric_b
        )

    # --------------------------------------------
    # 95% percentile confidence interval
    # --------------------------------------------
    ci_low, ci_high = np.percentile(
        bootstrap_differences,
        [2.5, 97.5]
    )

    # --------------------------------------------
    # Two-sided bootstrap probability
    #
    # How often does bootstrap difference cross
    # the opposite side of zero?
    # --------------------------------------------
    bootstrap_mean_difference = (
        bootstrap_differences.mean()
    )

    bootstrap_sd = (
        bootstrap_differences.std(ddof=1)
    )

    return {

        "metric": metric,

        "model_A":
            observed_a,

        "model_B":
            observed_b,

        "difference_A_minus_B":
            observed_difference,

        "bootstrap_mean_difference":
            bootstrap_mean_difference,

        "bootstrap_SD":
            bootstrap_sd,

        "CI_low":
            ci_low,

        "CI_high":
            ci_high
    }


if RUN_BOOTSTRAP:
    rows, pooled_rows, class_rows = [], [], []
    pairs = [('imagenet/frozen','scratch/full'),('radimagenet/frozen','scratch/full'),('radimagenet/frozen','imagenet/frozen')]
    for cohort_name,args in cohort_inputs.items():
        oof_results = {f'{init}/{ft}':generate_oof(init,ft,*args) if RUN_INFERENCE else pd.read_csv(args[-1]/f'OOF_{init}_{ft}_subject_predictions.csv') for init,ft in OOF_CONDITIONS}
        for name,frame in oof_results.items():
            assert not frame.subject_id.duplicated().any()
            assert set(frame.subject_id)==set(args[1].subject_id)
            expected_fold={sid:i for i,f in enumerate(args[2],1) for sid in f['test']}
            assert frame.fold.eq(frame.subject_id.map(expected_fold)).all()
            expected_labels=args[1].set_index('subject_id').label.loc[frame.subject_id].to_numpy()
            assert np.array_equal(frame.true_label.to_numpy(),expected_labels)
            prob=frame[[f'prob_class_{c}' for c in range(3)]].to_numpy()
            assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1,atol=1e-5)
            assert np.array_equal(frame.pred_label.to_numpy(),prob.argmax(axis=1))
            for metric in ['macro_f1','macro_auc','balanced_accuracy']:
                pooled_rows.append(dict(cohort=cohort_name,model=name,metric=metric,value=calculate_metric(frame.true_label,frame.pred_label,prob,metric)))
            report=classification_report(frame.true_label,frame.pred_label,labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
            for label in CLASS_NAMES:class_rows.append(dict(cohort=cohort_name,model=name,label=label,**report[label]))
            pd.DataFrame(confusion_matrix(frame.true_label,frame.pred_label,labels=[0,1,2]),index=CLASS_NAMES,columns=CLASS_NAMES).to_csv(args[-1]/f'confusion_{name.replace("/","_")}.csv')
        for a,b in pairs:
            for metric in ['macro_f1','macro_auc','balanced_accuracy']:
                result=paired_stratified_bootstrap(oof_results[a],oof_results[b],metric,n_boot=N_BOOT,seed=SEED)
                rows.append(dict(cohort=cohort_name,model_a=a,model_b=b,**result))
    stats_df=pd.DataFrame(rows)
    stats_df.to_csv(OUTPUT_DIR/'paired_bootstrap.csv',index=False)
    pd.DataFrame(pooled_rows).to_csv(OUTPUT_DIR/'pooled_metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT_DIR/'per_class_metrics.csv',index=False)
    display(stats_df[stats_df.metric=='macro_f1'])
else:
    print('Bootstrap calculations are off. Enable RUN_BOOTSTRAP after creating the OOF files.')


FIG_DIR=OUTPUT_DIR/'figures'
FIG_DIR.mkdir(exist_ok=True)
labels=[f'{a.title()} / {b.title()}' for a,b in CONDITIONS]
def save_figure(fig,name):
    fig.tight_layout()
    fig.savefig(FIG_DIR/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(FIG_DIR/f'{name}.pdf',bbox_inches='tight')
    plt.show()

full_path=OUTPUT_DIR/'full_results.csv'
age_path=AGE_OUTPUT_DIR/'full_results.csv'
if full_path.exists() and age_path.exists():
    frames=[pd.read_csv(full_path),pd.read_csv(age_path)]
    for frame in frames:summarize_folds(frame)
    for metric,title in [('test_subj_f1','Macro-F1'),('test_subj_auc','Macro-AUC')]:
        fig,ax=plt.subplots(figsize=(6.4,4.1))
        for frame,offset,label,color in zip(frames,[-.13,.13],['Full cohort (N=347)','Age ≥60 (N=166)'],['tab:blue','tab:orange']):
            summary=frame.groupby(['init','fine_tune'])[metric].agg(['mean','std']).reindex(CONDITIONS)
            ax.errorbar(summary['mean'],np.arange(7)+offset,xerr=summary['std'],fmt='o',capsize=3,label=label,color=color)
        ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.set_xlabel(title);ax.legend(fontsize=8)
        save_figure(fig,f'age_comparison_{metric}')
        fig,ax=plt.subplots(figsize=(6.4,4.1))
        summary=frames[0].groupby(['init','fine_tune'])[metric].agg(['mean','std']).reindex(CONDITIONS)
        ax.errorbar(summary['mean'],np.arange(7),xerr=summary['std'],fmt='o',capsize=3)
        ax.set_yticks(np.arange(7),labels);ax.invert_yaxis();ax.set_xlabel(title)
        save_figure(fig,f'full_{metric}')
else:
    print('Fold result files are not yet available; performance plots skipped.')
fig,ax=plt.subplots(figsize=(6.4,3.6))
groups=[audit.loc[audit.true_label==c,'Age'].dropna().to_numpy() for c in range(3)]
ax.boxplot(groups,positions=np.arange(3),showfliers=False)
rng=np.random.default_rng(SEED)
for i,values in enumerate(groups):ax.scatter(rng.normal(i,.055,len(values)),values,alpha=.25,s=12)
ax.axhline(60,ls='--',color='gray',label='Age ≥60 threshold')
ax.set_xticks(np.arange(3),['Non-Demented','Very Mild','Mild+Moderate']);ax.set_ylabel('Age (years)');ax.legend()
save_figure(fig,'age_distribution')
frac_path=OUTPUT_DIR/'ablation_results.csv'
if frac_path.exists():
    fractions=pd.read_csv(frac_path)
    summarize_folds(fractions)
    fig,ax=plt.subplots(figsize=(6.4,4))
    for (init,ft),group in fractions.groupby(['init','fine_tune']):
        summary=group.groupby('label_frac').test_subj_f1.agg(['mean','std'])
        ax.errorbar(summary.index,summary['mean'],yerr=summary['std'],marker='o',capsize=3,label=f'{init}/{ft}')
    ax.set_xlabel('Fraction of training subjects');ax.set_ylabel('Macro-F1');ax.legend()
    save_figure(fig,'label_fractions')
