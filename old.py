"""
class MnMs2MultiViewNiftiDataset(Dataset):
    """
    Reads M&Ms-2 per-ID folders with the 10 files:

      ID_LA_CINE.nii.gz
      ID_LA_ED_gt_.nii.gz
      ID_LA_ED.nii.gz
      ID_LA_ES_gt_.nii.gz
      ID_LA_ES.nii.gz
      ID_SA_CINE.nii.gz
      ID_SA_ED_gt_.nii.gz
      ID_SA_ED.nii.gz
      ID_SA_ES_gt_.nii.gz
      ID_SA_ES.nii.gz

    CSV is expected to have columns: ID,Class

    view_phase_list: list of strings like ["LA_ED", "SA_ED", "LA_ES", "SA_ES"].
    For each sample we pick one (view, phase) from this list.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        view_phase_list=None,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        if not {"SUBJECT_CODE", "DISEASE"}.issubset(self.df.columns):
            raise ValueError("CSV must contain SUBJECT_CODE and DISEASE columns.")

        bad_labels = set(self.df["DISEASE"]) - set(CARDIOMYOPATHY_LABELS)
        if bad_labels:
            raise ValueError(f"Unexpected labels found: {bad_labels}")

        # Default: single-view LA+ED
        if view_phase_list is None or len(view_phase_list) == 0:
            view_phase_list = ["LA_ED"]

        # Normalise config strings, e.g. "la_ed" -> ("LA","ED")
        self.configs = []
        for vp in view_phase_list:
            vp_norm = vp.strip().upper()
            if "_" not in vp_norm:
                raise ValueError(f"view_phase '{vp}' must be of form 'LA_ED' or 'SA_ES'")
            view, phase = vp_norm.split("_", 1)
            if view not in {"LA", "SA"}:
                raise ValueError(f"View must be 'LA' or 'SA', got '{view}'")
            if phase not in {"ED", "ES"}:
                raise ValueError(f"Phase must be 'ED' or 'ES', got '{phase}'")
            self.configs.append((view, phase))

    def __len__(self):
        return len(self.df)

    def _volume_path(self, case_id: str, view: str, phase: str) -> str:
        # Example: "123_LA_ED.nii.gz" inside img_root/123/
        filename = f"{case_id}_{view}_{phase}.nii.gz"
        #print(f"Looking for volume at: {filename}")
        #a
        return os.path.join(self.img_root, case_id, filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = pad_subject_code(row["SUBJECT_CODE"])
        label_str = row["DISEASE"]
        label = LABEL_TO_IDX[label_str]

        # Pick one (view, phase) config – controlled via view_phase_list
        view, phase = random.choice(self.configs)

        vol_path = self._volume_path(case_id, view, phase)
        img_nii = nib.load(vol_path)
        vol = img_nii.get_fdata()  # typically (H, W, S)

        # Take mid slice along last axis; easy to change later
        mid_slice_idx = vol.shape[-1] // 2
        slice2d = vol[..., mid_slice_idx]

        # Intensity normalisation to [0, 255]
        slice2d = slice2d - slice2d.min()
        if slice2d.max() > 0:
            slice2d = slice2d / slice2d.max()
        slice2d = (slice2d * 255).astype(np.uint8)

        im = Image.fromarray(slice2d).convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        return im, label

def pad_subject_code(code) -> str:
    return str(code).zfill(3)   # 1 -> "001", 12 -> "012", 123 -> "123"

def get_mnms2_dataloaders(
    train_csv: str,
    test_csv: str,
    img_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    view_phase_list=None,   # e.g. ["LA_ED"] or ["LA_ED","SA_ED","LA_ES","SA_ES"]
    val_frac: float = 0.1,
    seed: int = 42,
):
    """
    Build train/val/test loaders from train+test CSVs.
    A val subset is carved out from the train set with random_split.
    """
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Full train dataset (before val split)
    train_full_ds = MnMs2MultiViewNiftiDataset(
        train_csv, img_root, view_phase_list=view_phase_list, transform=transform
    )

    n_full = len(train_full_ds)
    val_size = max(1, int(val_frac * n_full))
    train_size = n_full - val_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full_ds, [train_size, val_size], generator=gen)

    # Test dataset
    test_ds = MnMs2MultiViewNiftiDataset(
        test_csv, img_root, view_phase_list=view_phase_list, transform=transform
    )

    # Class weights from the train subset
    # Use indices of train_ds into train_full_ds.df
    full_df = train_full_ds.df
    train_indices = train_ds.indices  # list of indices into full_df
    labels = [LABEL_TO_IDX[full_df["DISEASE"].iloc[i]] for i in train_indices]
    counts = np.bincount(labels, minlength=len(CARDIOMYOPATHY_LABELS))
    class_weights = 1.0 / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(CARDIOMYOPATHY_LABELS), class_weights
"""