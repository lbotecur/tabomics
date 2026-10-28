# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Augmentations of a batch of prior tables during training: widening, sparse columns and missing values.

Synthetic tables have tens of columns; omics tables have thousands, strongly correlated, few of them informative.
Widening adds columns built from the existing ones, following Algorithms 1 and 2 of TabPFN-Wide (Kolberg et al.):
continuous columns as sparse linear combinations of the continuous ones plus noise, categorical columns by mixing
the categorical ones. Sparse columns imitate bag-of-words or count data (mostly zeros). Missing values are set
completely at random, so that the model learns to use its missing-value indicator.

`augment_batch` applies all of them to a batch as the training loop does, drawing from torch's global generator
(or the one given) in a fixed order, so the same seed gives the same augmented tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Where widening takes the categorical columns from (see AugmentationConfig.widening_categorical).
WIDENING_CATEGORICAL = ("prior", "detect")


@dataclass
class AugmentationConfig:
    """How the training tables are augmented (everything off by default).

    Attributes:
        add_features_min: Fewest columns added to a widened batch.
        add_features_max: Most columns added to a widened batch; 0 disables widening.
        sparsity_max: Largest share of existing columns combined into a new one (drawn in [0, max] per batch;
            TabPFN-Wide uses 0.05).
        noise_max: Largest noise scale of the new continuous columns (drawn in [0, max] per batch).
        include_original_prob: Probability that a widened batch keeps its original columns (shuffled in).
        max_cats: Most categories of a new categorical column.
        prob_no_widening: Share of batches left narrow.
        widen_max_rows: Rows kept in a widened batch (0 = all), so that thousands of columns fit in memory; narrow
            batches keep all their rows.
        sparse_columns_prob: Probability that a widened batch gets sparse columns.
        sparse_columns_max_fraction: Largest share of its columns made sparse (drawn in [0, max]).
        missing_rate_max: Largest share of cells set missing (drawn in [0, max] per batch); 0 disables it.
        widening_categorical: Where widening takes the categorical columns from. "prior": the prior's categorical
            masks when it gives them, the new categorical columns recoded 0..K-1. "detect": columns with at most 20
            distinct values (TabPFN-Wide's rule), not recoded; the recipes of v2 and v2.5 use it, as their original
            runs did. Without masks from the prior, both detect.
    """

    add_features_min: int = 0
    add_features_max: int = 0
    sparsity_max: float = 0.05
    noise_max: float = 1.0
    include_original_prob: float = 0.5
    max_cats: int = 20
    prob_no_widening: float = 0.0
    widen_max_rows: int = 0
    sparse_columns_prob: float = 0.0
    sparse_columns_max_fraction: float = 1.0
    missing_rate_max: float = 0.0
    widening_categorical: str = "prior"

    def __post_init__(self):
        if self.widening_categorical not in WIDENING_CATEGORICAL:
            raise ValueError(
                f"widening_categorical must be one of {WIDENING_CATEGORICAL}, got {self.widening_categorical!r}"
            )


def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: int,
    config: AugmentationConfig,
    categorical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Widens, sparsifies and blanks a batch of tables as configured.

    Random draws, from torch's global generators, in this order: whether to widen, how many columns, sparsity,
    noise, the widening itself, whether to add sparse columns and how many, the missing-value rate and cells.

    Args:
        x: Features, shape (B, T, H).
        y: Targets, shape (B, T).
        train_size: Training rows (the first ones).
        config: The augmentations.
        categorical: Categorical columns, shape (B, H), if the prior gives them (their codes are 0..K-1, and the
            new categorical columns get such codes too); None = detected (at most 20 distinct values). Ignored with
            config.widening_categorical = "detect".

    Returns:
        The new features (B, T', H'), targets (B, T') and training rows: T' < T only if a widened batch is
        shortened.
    """
    widen = config.add_features_max > 0 and (
        config.prob_no_widening == 0 or torch.rand(1).item() >= config.prob_no_widening
    )
    if widen and config.widen_max_rows and x.shape[1] > config.widen_max_rows:
        x, y, train_size = shorten_table(x, y, train_size, config.widen_max_rows)
    if widen:
        num_add = int(torch.randint(config.add_features_min, config.add_features_max + 1, (1,)).item())
        sparsity = torch.rand(1).item() * config.sparsity_max
        noise = torch.rand(1).item() * config.noise_max
        masks = categorical if config.widening_categorical == "prior" else None
        x = add_mixed_widening_features(
            x,
            num_add,
            sparsity,
            noise,
            max_cats=config.max_cats,
            include_original_prob=config.include_original_prob,
            categorical_masks=masks,
            recode_categoricals=masks is not None,
        )
        if config.sparse_columns_prob > 0 and torch.rand(1).item() < config.sparse_columns_prob:
            x = sparsify_columns(x, torch.rand(1).item() * config.sparse_columns_max_fraction)
    if config.missing_rate_max > 0:
        rate = torch.rand(1, device=x.device).item() * config.missing_rate_max
        x = inject_mcar_missingness(x, rate)
    return x, y, train_size


def shorten_table(x: torch.Tensor, y: torch.Tensor, train_size: int, max_rows: int):
    """Keeps max_rows rows of a batch, with the same share of training rows (at least 2) and test rows (at least 1).

    The prior draws rows independently, so the first rows of each part are a random subset of it.

    Args:
        x: Features, shape (B, T, H).
        y: Targets, shape (B, T).
        train_size: Training rows.
        max_rows: Rows to keep.

    Returns:
        The shortened features and targets, and their number of training rows.
    """
    rows = x.shape[1]
    n_train = min(max(2, round(train_size * max_rows / rows)), train_size, max_rows - 1)
    n_test = min(max_rows - n_train, rows - train_size)
    keep = torch.cat([torch.arange(n_train), torch.arange(train_size, train_size + n_test)])
    return x[:, keep.to(x.device)], y[:, keep.to(y.device)], n_train


def inject_mcar_missingness(x: torch.Tensor, rate: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sets each cell to NaN independently with probability `rate` (missing completely at random).

    Features only, never targets. The model fills missing values and flags them, so this teaches it to use the
    flag.

    Args:
        x: Features (any shape).
        rate: Probability of each cell being missing, in [0, 1].
        generator: Random generator (torch's global one if None).

    Returns:
        A new tensor.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}.")
    x = x.clone()
    if rate == 0.0:
        return x
    mask = torch.rand(x.shape, generator=generator, device=x.device) < rate
    x[mask] = float("nan")
    return x


def add_widening_features(
    x: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    noise_std: float,
    include_original_prob: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Continuous widening (Algorithm 1 of TabPFN-Wide).

    Each new column is a sparse random linear combination of the existing ones plus Gaussian noise proportional to
    its own standard deviation over the rows, which gives the strongly correlated columns of omics data. With
    probability `include_original_prob` the original columns are kept and all columns shuffled.

    Args:
        x: Features, shape (..., rows, columns).
        num_features_to_add: New columns (>= 0).
        sparsity: Probability that a weight of the combination is nonzero.
        noise_std: Noise scale.
        include_original_prob: Probability of keeping the original columns.
        generator: Random generator (torch's global one if None).

    Returns:
        A new tensor of shape (..., rows, new columns [+ original columns]).
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")
    if num_features_to_add == 0:
        return x.clone()

    m = x.shape[-1]
    weights = torch.randn(m, num_features_to_add, generator=generator, device=x.device, dtype=x.dtype)
    mask = (torch.rand(m, num_features_to_add, generator=generator, device=x.device) < sparsity).to(x.dtype)
    x_wide = x @ (mask * weights)

    stds = x_wide.std(dim=-2, keepdim=True)
    stds = torch.where(stds == 0, torch.ones_like(stds), stds)  # an all-zero combination: pure noise
    noise = torch.randn(x_wide.shape, generator=generator, device=x.device, dtype=x.dtype) * (noise_std * stds)
    x_wide = x_wide + noise

    if torch.rand(1, generator=generator, device=x.device).item() < include_original_prob:
        x_wide = torch.cat([x, x_wide], dim=-1)
        x_wide = x_wide[..., torch.randperm(x_wide.shape[-1], generator=generator, device=x.device)]
    return x_wide


def detect_categorical_mask(x: torch.Tensor, max_unique: int = 20) -> torch.Tensor:
    """Columns with at most `max_unique` distinct observed values, taken as categorical (TabPFN-Wide's rule).

    Only for priors that do not say which columns are categorical. Missing values are not counted as a value.

    Args:
        x: Features of one table, shape (rows, columns).
        max_unique: Most distinct values of a categorical column.

    Returns:
        Boolean tensor of shape (columns,).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (num_samples, num_features), got shape {tuple(x.shape)}.")
    # sorted columns, NaN last: a new value starts where two consecutive observed values differ
    x_sorted, _ = torch.sort(x, dim=0)
    nan_sorted = torch.isnan(x_sorted)
    both_valid = (~nan_sorted[:-1]) & (~nan_sorted[1:])
    new_value = both_valid & (x_sorted[:-1] != x_sorted[1:])
    has_valid = (~nan_sorted).any(dim=0)
    return new_value.sum(dim=0) + has_valid.long() <= max_unique


def split_widening_budget(cat_mask: torch.Tensor, num_features_to_add: int) -> tuple[int, int]:
    """Splits the new columns between continuous and categorical ones in the table's proportion.

    Args:
        cat_mask: Categorical columns of the table, shape (columns,).
        num_features_to_add: New columns (>= 0).

    Returns:
        (continuous, categorical) new columns; the categorical count is rounded down.
    """
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    num_total = cat_mask.numel()
    if num_total == 0:
        raise ValueError("cat_mask must not be empty.")
    num_cat_to_add = int(int(cat_mask.sum().item()) / num_total * num_features_to_add)
    return num_features_to_add - num_cat_to_add, num_cat_to_add


def sample_target_cardinalities(
    size: int, max_cats: int, low: int = 3, lam: float = 0.08, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Numbers of categories of new categorical columns, from a discrete exponential on [low, max_cats].

    As the prose of TabPFN-Wide (and its implementation) says; its pseudocode says uniform.

    Args:
        size: How many (>= 0).
        max_cats: Largest number of categories (>= 1).
        low: Smallest number of categories (lowered to max_cats if larger).
        lam: Rate of the exponential: larger favors fewer categories.
        generator: Random generator (torch's global one if None).

    Returns:
        Long tensor of shape (size,).
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if max_cats < 1:
        raise ValueError(f"max_cats must be >= 1, got {max_cats}.")
    device = generator.device if generator is not None else torch.device("cpu")
    if size == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    low = min(low, max_cats)
    values = torch.arange(low, max_cats + 1, device=device)
    weights = torch.exp(-lam * (values - low).float())
    idx = torch.multinomial(weights / weights.sum(), num_samples=size, replacement=True, generator=generator)
    return values[idx]


def reduce_cardinality(col: torch.Tensor, max_cats: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Keeps the `max_cats` most frequent categories of a column and maps each rarer one to a random kept one.

    The end state of TabPFN-Wide's loop that merges the rarest category into a frequent one (Algorithm 2).

    Args:
        col: One categorical column without missing values.
        max_cats: Categories to keep (>= 1).
        generator: Random generator (torch's global one if None).

    Returns:
        A new column with at most `max_cats` distinct values.
    """
    if max_cats < 1:
        raise ValueError(f"max_cats must be >= 1, got {max_cats}.")
    vals, inv = torch.unique(col, return_inverse=True)
    num_categories = vals.numel()
    if num_categories <= max_cats:
        return col.clone()
    order = torch.argsort(torch.bincount(inv, minlength=num_categories))  # rarest first
    head, tail = order[-max_cats:], order[:-max_cats]
    target = torch.randint(0, head.numel(), (tail.numel(),), generator=generator, device=col.device)
    mapping = torch.arange(num_categories, device=col.device)
    mapping[tail] = head[target]
    return vals[mapping[inv]]


def add_categorical_widening_features(
    x_cat: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    max_cats: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Categorical widening of one table (Algorithm 2 of TabPFN-Wide).

    Each new column takes, in every row, the value of one of k random donor columns (k = max(1, sparsity x
    categorical columns)), and is then reduced to a random number of categories.

    Args:
        x_cat: Categorical columns of one table, shape (rows, categorical columns).
        num_features_to_add: New columns (>= 0).
        sparsity: Share of the categorical columns used as donors.
        max_cats: Most categories of a new column.
        generator: Random generator (torch's global one if None).

    Returns:
        Tensor of shape (rows, num_features_to_add).
    """
    if x_cat.ndim != 2:
        raise ValueError(f"x_cat must be 2D (num_samples, num_categorical), got {tuple(x_cat.shape)}.")
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    num_samples, num_categorical = x_cat.shape
    if num_features_to_add == 0:
        return x_cat.new_empty((num_samples, 0))
    if num_categorical == 0:
        raise ValueError("Cannot generate categorical features without at least one donor column.")

    k = max(1, int(sparsity * num_categorical))
    target_cards = sample_target_cardinalities(num_features_to_add, max_cats, generator=generator)
    rows = torch.arange(num_samples, device=x_cat.device)
    out = x_cat.new_empty((num_samples, num_features_to_add))
    for j in range(num_features_to_add):
        donors = torch.randperm(num_categorical, generator=generator, device=x_cat.device)[:k]
        donor_per_row = torch.randint(0, k, (num_samples,), generator=generator, device=x_cat.device)
        out[:, j] = reduce_cardinality(x_cat[rows, donors[donor_per_row]], int(target_cards[j]), generator=generator)
    return out


def add_mixed_widening_features(
    x: torch.Tensor,
    num_features_to_add: int,
    sparsity: float,
    noise_std: float,
    max_cats: int = 20,
    include_original_prob: float = 0.5,
    max_unique_for_categorical: int = 20,
    generator: torch.Generator | None = None,
    categorical_masks: torch.Tensor | None = None,
    recode_categoricals: bool = False,
) -> torch.Tensor:
    """Widens every table of a batch with new continuous and categorical columns (TabPFN-Wide).

    Per table: the new columns are split between continuous (Algorithm 1) and categorical (Algorithm 2) in the
    table's proportion. Once for the whole batch (so that every table keeps the same width), with probability
    `include_original_prob`, the original columns are kept and each table's columns shuffled.

    Args:
        x: Features, shape (B, rows, columns).
        num_features_to_add: New columns per table (>= 0).
        sparsity: Sparsity of both algorithms.
        noise_std: Noise scale of the continuous columns.
        max_cats: Most categories of a new categorical column.
        include_original_prob: Probability of keeping the original columns.
        max_unique_for_categorical: Detection threshold when the categorical columns are not given.
        generator: Random generator (torch's global one if None).
        categorical_masks: Categorical columns of each table, shape (B, columns); None = detected.
        recode_categoricals: Give each new categorical column the codes 0..K-1 in a random order, as an ordinal
            encoder does at inference (for priors whose categorical columns are such codes).

    Returns:
        A new tensor of shape (B, rows, num_features_to_add [+ columns]).
    """
    if x.ndim != 3:
        raise ValueError(f"x must be 3D (batch_size, num_samples, num_features), got {tuple(x.shape)}.")
    if num_features_to_add < 0:
        raise ValueError(f"num_features_to_add must be >= 0, got {num_features_to_add}.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}.")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0, got {noise_std}.")
    if num_features_to_add == 0:
        return x.clone()

    include_original = torch.rand(1, generator=generator, device=x.device).item() < include_original_prob
    widened = []
    for b in range(x.shape[0]):
        xb = x[b]
        if categorical_masks is not None:
            cat_mask = categorical_masks[b].to(device=x.device, dtype=torch.bool)
        else:
            cat_mask = detect_categorical_mask(xb, max_unique=max_unique_for_categorical)
        num_cont_to_add, num_cat_to_add = split_widening_budget(cat_mask, num_features_to_add)

        blocks = []
        if num_cont_to_add > 0:
            blocks.append(
                add_widening_features(
                    xb[:, ~cat_mask], num_cont_to_add, sparsity, noise_std, include_original_prob=0.0,
                    generator=generator,
                )
            )  # fmt: skip
        if num_cat_to_add > 0:
            new_cat = add_categorical_widening_features(
                xb[:, cat_mask], num_cat_to_add, sparsity, max_cats=max_cats, generator=generator
            )
            if recode_categoricals:
                new_cat = torch.stack([recode_categories(c, generator) for c in new_cat.T], dim=1)
            blocks.append(new_cat)
        new_features = torch.cat(blocks, dim=-1)

        if include_original:
            new_features = torch.cat([xb, new_features], dim=-1)
            perm = torch.randperm(new_features.shape[-1], generator=generator, device=x.device)
            new_features = new_features[:, perm]
        widened.append(new_features)
    return torch.stack(widened, dim=0)


def recode_categories(col: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Maps the K distinct values of a categorical column to the codes 0..K-1 in a random order.

    An ordinal encoder's codes are in an arbitrary order too (e.g. alphabetical). Missing values stay missing.

    Args:
        col: One categorical column.
        generator: Random generator (torch's global one if None).

    Returns:
        A new column.
    """
    out = col.clone()
    present = ~torch.isnan(col)
    values, inverse = torch.unique(col[present], return_inverse=True)
    codes = torch.randperm(len(values), generator=generator).to(col.device)
    out[present] = codes[inverse].to(col.dtype)
    return out


def sparsify_columns(
    x: torch.Tensor,
    fraction: float,
    min_nonzero: float = 0.002,
    max_nonzero: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Turns a share of the columns into sparse ones, as bag-of-words text or single-cell counts.

    Each chosen column keeps only its highest values, shifted to start at 0, and is 0 elsewhere. Its share of
    nonzero rows is drawn log-uniformly in [min_nonzero, max_nonzero], so some values are rare (a value present in
    a share p of the rows becomes z ~ sqrt((1 - p) / p) once standardized). A nonzero entry keeps the column's
    information (it is above a threshold of the original value), and a zero is informative too (the absence of a
    word). Missing values stay missing.

    Args:
        x: Features, shape (B, rows, columns).
        fraction: Probability that each column is made sparse.
        min_nonzero: Smallest share of nonzero rows of a sparse column.
        max_nonzero: Largest share of nonzero rows.
        generator: Random generator (torch's global one if None).

    Returns:
        A new tensor.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}.")
    b, _, f = x.shape
    out = x.clone()
    chosen = torch.rand(b, f, generator=generator) < fraction
    log_lo, log_hi = torch.log(torch.tensor(min_nonzero)), torch.log(torch.tensor(max_nonzero))
    keep = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(b, f, generator=generator))  # share of nonzero rows
    for i in range(b):
        cols = torch.nonzero(chosen[i]).flatten()
        if len(cols) == 0:
            continue
        values = x[i][:, cols].float()
        # each column's (1 - keep) quantile, ignoring missing values (sorted last)
        ordered = torch.sort(torch.nan_to_num(values, nan=float("inf")), dim=0).values
        observed = (~torch.isnan(values)).sum(dim=0).clamp(min=1)
        position = ((1 - keep[i, cols].to(values.device)) * (observed - 1)).floor().long()
        thresholds = ordered.gather(0, position[None]).squeeze(0)
        sparse = torch.where(values > thresholds, values - thresholds, torch.zeros_like(values))
        sparse = torch.where(torch.isnan(values), values, sparse)
        out[i][:, cols] = sparse.to(x.dtype)
    return out
