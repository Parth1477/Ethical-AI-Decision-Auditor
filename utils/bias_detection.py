"""
bias_detection.py — Advanced Bias Detection Engine v3
======================================================
New in v3:
  - AUTOMATIC sensitive attribute detection using 2-stage heuristics
    (dtype + cardinality + exclusion rules) instead of keyword-only matching
  - Detection reason metadata returned per attribute
  - Numeric low-cardinality columns treated as categorical groups
  - All print() replaced with logger calls (Windows encoding-safe)
  - ZeroDivisionError and NaN guards throughout
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── Decision Label Vocabulary ─────────────────────────────────────────────────
POSITIVE_LABELS = {
    'selected', 'approved', 'accepted', 'hired', 'passed',
    'granted', 'promoted', 'admitted', 'yes', '1', 'true', 'positive'
}
NEGATIVE_LABELS = {
    'rejected', 'denied', 'declined', 'fired', 'failed',
    'revoked', 'dismissed', 'refused', 'no', '0', 'false', 'negative'
}

# ── Columns to Exclude from Sensitive Attribute Candidates ────────────────────
EXCLUDE_KEYWORDS = [
    'id', 'name', 'index', 'timestamp', 'date', 'row',
    'unnamed', 'serial', 'uuid', 'email', 'phone', 'address',
    'salary', 'income', 'wage', 'pay', 'score', 'grade',
    'experience', 'years', 'gpa', 'credit', 'account', 'amount',
]

# ── Keyword Hint: column names containing these get detection_reason="keyword" ─
PROTECTED_KEYWORDS = [
    'gender', 'sex', 'race', 'ethnicity', 'age', 'religion',
    'nationality', 'disability', 'marital', 'orientation', 'caste',
    'tribe', 'color', 'colour', 'region', 'department', 'education',
    'degree', 'qualification', 'language', 'citizen', 'birth',
]

# ── Cardinality Thresholds ────────────────────────────────────────────────────
MAX_CATEGORICAL_UNIQUE    = 20   # object/category dtype: keep if <= 20 unique values
MAX_NUMERIC_ENCODED_UNIQUE = 10  # numeric dtype: treat as categorical if <= 10 unique values
HIGH_CARDINALITY_RATIO    = 0.4  # if nunique/nrows > this → likely an ID/continuous


def _age_band(age):
    """Bucket continuous Age into readable bands for DI analysis."""
    try:
        age = int(float(str(age).strip()))
    except (ValueError, TypeError):
        return "Unknown"
    if age < 30:
        return "Under 30"
    elif age <= 45:
        return "30-45"
    else:
        return "Over 45"


def _normalize_decision(value, positive_label, negative_label):
    """Map a raw decision value to 1 (positive) or 0 (negative)."""
    raw = str(value).strip().lower()
    if raw == positive_label:
        return 1
    if raw == negative_label:
        return 0
    if raw in POSITIVE_LABELS:
        return 1
    if raw in NEGATIVE_LABELS:
        return 0
    return 0


def _detect_decision_labels(series):
    """
    Auto-detect positive and negative labels in a decision column.
    Returns (positive_label, negative_label, unique_values_list).
    """
    counts = series.astype(str).str.strip().str.lower().value_counts()
    unique = list(counts.index)

    if len(unique) == 0:
        return 'selected', 'rejected', []
    if len(unique) == 1:
        return unique[0], '__none__', unique

    positive_found = [u for u in unique if u in POSITIVE_LABELS]
    negative_found = [u for u in unique if u in NEGATIVE_LABELS]

    if positive_found and negative_found:
        logger.info("Decision labels from vocabulary: pos=%r neg=%r", positive_found[0], negative_found[0])
        return positive_found[0], negative_found[0], unique

    logger.info("Decision labels by frequency: pos=%r neg=%r", unique[0], unique[1])
    return unique[0], unique[1] if len(unique) > 1 else '__none__', unique


def _is_excluded(col_name):
    """Return True if column should be excluded from sensitive attribute analysis."""
    col_lower = col_name.strip().lower()
    for kw in EXCLUDE_KEYWORDS:
        if kw in col_lower:
            return True
    return False


def _has_protected_keyword(col_name):
    """Return the matched keyword if column name suggests a protected attribute, else None."""
    col_lower = col_name.strip().lower()
    for kw in PROTECTED_KEYWORDS:
        if kw in col_lower:
            return kw
    return None


def detect_sensitive_attributes(df, decision_col):
    """
    Two-stage heuristic sensitive attribute detection.

    Stage 1 — Exclusions:
      - Decision column itself
      - Columns with EXCLUDE_KEYWORDS (ID, Salary, Income, etc.)
      - Columns where ALL values are unique (likely an ID column)
      - Numeric columns with high cardinality ratio (> HIGH_CARDINALITY_RATIO)

    Stage 2 — Inclusions:
      - object/category dtype with nunique <= MAX_CATEGORICAL_UNIQUE
      - numeric dtype with nunique <= MAX_NUMERIC_ENCODED_UNIQUE
        (encoded categoricals like 0/1 for Gender)

    Age-named numeric columns are auto-bucketed into bands.

    Returns:
        list of dicts:
          {
            'col': column name,
            'series': processed Series (bucketed if numeric age),
            'reason': 'keyword' | 'categorical' | 'numeric_categorical',
            'n_unique': int,
            'dtype': str,
          }
    """
    results = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        # ── Skip the decision column ───────────────────────────────────────────
        if col.strip().lower() == decision_col.strip().lower():
            continue
        if col == decision_col:
            continue

        # ── Stage 1: Exclusion rules ───────────────────────────────────────────
        if _is_excluded(col):
            logger.debug("Excluded column '%s' (exclusion keyword match).", col)
            continue

        series = df[col].dropna()
        n_unique = series.nunique()
        dtype_str = str(df[col].dtype)

        # All-unique → likely ID
        if n_unique == n_rows and n_rows > 10:
            logger.debug("Excluded column '%s' (all unique — likely ID).", col)
            continue

        # ── Stage 2: Inclusion rules ───────────────────────────────────────────
        keyword = _has_protected_keyword(col)
        # Pandas 3.x / Python 3.13+: string columns may report dtype as pd.StringDtype
        # ('str') rather than old object dtype ('O'/'object').
        # Use is_string_dtype for forward-compatible detection of string/categorical cols.
        dtype_name = str(df[col].dtype).lower()
        is_object = (
            df[col].dtype == object           # legacy object dtype
            or dtype_name in ('object', 'str', 'string', 'category')
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col]) if hasattr(pd.api.types, 'is_categorical_dtype') else False
        )
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) and not is_object

        # Object/category type with low cardinality
        if is_object and n_unique <= MAX_CATEGORICAL_UNIQUE:
            reason = 'keyword' if keyword else 'categorical'
            processed = df[col].astype(str).str.strip()
            logger.info("Sensitive attribute detected: '%s' (reason=%s, n_unique=%d)", col, reason, n_unique)
            results.append({
                'col': col, 'series': processed, 'reason': reason,
                'n_unique': n_unique, 'dtype': dtype_str,
            })
            continue

        # Numeric type
        if is_numeric:
            cardinality_ratio = n_unique / n_rows
            if cardinality_ratio > HIGH_CARDINALITY_RATIO and not keyword:
                logger.debug("Excluded numeric column '%s' (high cardinality ratio=%.2f).", col, cardinality_ratio)
                continue

            # Age-like column → band into groups
            if keyword and 'age' in keyword:
                processed = df[col].apply(_age_band)
                reason = 'keyword_age_banded'
                logger.info("Sensitive attribute '%s' detected as age-banded (reason=%s).", col, reason)
                results.append({
                    'col': col, 'series': processed, 'reason': reason,
                    'n_unique': processed.nunique(), 'dtype': dtype_str,
                })
                continue

            # Low-cardinality numeric → treat as categorical (e.g. 0/1/2 encoding)
            if n_unique <= MAX_NUMERIC_ENCODED_UNIQUE:
                reason = 'keyword' if keyword else 'numeric_categorical'
                processed = df[col].astype(str).str.strip()
                logger.info("Sensitive attribute '%s' detected as numeric categorical (n_unique=%d, reason=%s).",
                            col, n_unique, reason)
                results.append({
                    'col': col, 'series': processed, 'reason': reason,
                    'n_unique': n_unique, 'dtype': dtype_str,
                })
                continue

    if not results:
        # Final fallback: use the first non-decision non-excluded column
        for col in df.columns:
            if col.strip().lower() != decision_col.strip().lower() and not _is_excluded(col):
                logger.warning("No sensitive attributes auto-detected — using fallback column: '%s'.", col)
                results.append({
                    'col': col,
                    'series': df[col].astype(str).str.strip(),
                    'reason': 'fallback',
                    'n_unique': df[col].nunique(),
                    'dtype': str(df[col].dtype),
                })
                break

    logger.info("detect_sensitive_attributes complete. Found %d attribute(s): %s",
                len(results), [r['col'] for r in results])
    return results


def _plain_english_explanation(attribute, group_rates, di, parity_diff, detection_reason=''):
    """Generate a human-readable bias explanation for an attribute."""
    if not group_rates:
        return ""

    sorted_groups = sorted(group_rates.items(), key=lambda x: x[1])
    min_group, min_rate = sorted_groups[0]
    max_group, max_rate = sorted_groups[-1]

    reason_label = {
        'keyword': 'Protected attribute (keyword match)',
        'categorical': 'Auto-detected categorical column',
        'numeric_categorical': 'Auto-detected encoded categorical column',
        'keyword_age_banded': 'Age-banded protected attribute',
        'fallback': 'Fallback attribute (no others detected)',
    }.get(detection_reason, 'Detected attribute')

    lines = [
        f"Attribute: {attribute} | Detection: {reason_label}",
        "",
        f"- {max_group} selection rate: {max_rate*100:.1f}%",
        f"- {min_group} selection rate: {min_rate*100:.1f}%",
        "",
    ]

    if di < 0.8:
        threshold_pct = max_rate * 0.8 * 100
        lines += [
            f"VIOLATES 4/5ths rule: {min_rate*100:.1f}% < "
            f"{threshold_pct:.1f}% (80% of {max_rate*100:.1f}%).",
            "",
            f"Disparate Impact Ratio: {di:.2f} (threshold >= 0.80)",
            f"Demographic Parity Difference: {parity_diff*100:.1f}%",
        ]
    else:
        lines += [
            f"Satisfies 4/5ths rule (DI = {di:.2f} >= 0.80).",
            f"Demographic Parity Difference: {parity_diff*100:.1f}%",
        ]

    return "\n".join(lines)


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_bias(dataset_path_or_df):
    """
    Analyze a dataset for bias across all detected sensitive attributes.

    Returns dict with:
        bias_detected       bool
        severity_score      float 0-100
        decision_labels     dict
        detected_attributes list of {col, reason, n_unique, dtype}
        metrics             dict {disparate_impact, selection_rates, demographic_parity_diff}
        bias_explanations   list of per-attribute explanation dicts
    """
    logger.info("=" * 60)
    logger.info("analyze_bias called.")

    # ── Load data ──────────────────────────────────────────────────────────────
    if isinstance(dataset_path_or_df, str):
        df = pd.read_csv(dataset_path_or_df)
        logger.info("CSV loaded: %s | rows=%d cols=%d", dataset_path_or_df, len(df), len(df.columns))
    else:
        df = dataset_path_or_df.copy()
        logger.info("DataFrame received | rows=%d cols=%d", len(df), len(df.columns))

    df.columns = df.columns.str.strip()
    logger.debug("Dataset shape: %s", df.shape)

    # ── Identify decision column ───────────────────────────────────────────────
    decision_col = None
    for col in df.columns:
        if col.strip().lower() == 'decision':
            decision_col = col
            break
    if decision_col is None:
        decision_col = df.columns[-1]
        logger.warning("No 'Decision' column found — using last column: '%s'", decision_col)

    logger.info("Decision column: '%s'", decision_col)

    # ── Detect and normalize decision labels ───────────────────────────────────
    pos_label, neg_label, all_labels = _detect_decision_labels(df[decision_col])
    logger.info("Decision labels: pos=%r neg=%r all=%s", pos_label, neg_label, all_labels)

    # ── Auto-detect sensitive attributes BEFORE adding outcome column ──────────
    # IMPORTANT: must run on clean df so the synthetic 'outcome' column (0/1) is
    # never mistaken for a low-cardinality categorical sensitive attribute.
    sensitive_attrs = detect_sensitive_attributes(df, decision_col)
    logger.info("Sensitive attributes [%d]: %s", len(sensitive_attrs), [a['col'] for a in sensitive_attrs])

    # ── Normalise decision column into numeric outcome ─────────────────────────
    df['outcome'] = df[decision_col].apply(
        lambda x: _normalize_decision(x, pos_label, neg_label)
    )

    overall_rate = df['outcome'].mean()
    logger.info("Overall positive rate: %.1f%%", overall_rate * 100)

    # ── Initialize output structures ───────────────────────────────────────────
    metrics = {
        'disparate_impact': {},
        'selection_rates': {},
        'demographic_parity_diff': {},
    }
    bias_explanations = []
    per_attr_severity = []
    bias_detected = False

    # ── Per-attribute analysis ─────────────────────────────────────────────────
    for attr_info in sensitive_attrs:
        attr_name       = attr_info['col']
        attr_series     = attr_info['series']
        detection_reason = attr_info['reason']

        temp = df.copy()
        temp['_attr'] = attr_series.values

        # Selection rates per group
        rates = temp.groupby('_attr')['outcome'].mean().to_dict()
        rates = {str(k): round(float(v), 4) for k, v in rates.items() if not pd.isna(v)}
        metrics['selection_rates'][attr_name] = rates
        logger.info("Selection rates [%s]: %s", attr_name, rates)

        if len(rates) < 2:
            logger.info("Skipping DI for '%s' — fewer than 2 groups.", attr_name)
            bias_explanations.append({
                'attribute': attr_name,
                'group_rates': rates,
                'di_ratio': 1.0,
                'parity_diff': 0.0,
                'bias_detected': False,
                'detection_reason': detection_reason,
                'explanation_text': f"'{attr_name}' has only one group — DI cannot be calculated.",
            })
            continue

        vals = list(rates.values())
        min_rate = min(vals)
        max_rate = max(vals)

        # Disparate Impact
        if max_rate == 0:
            di = 1.0
            logger.warning("All outcomes negative for '%s' — DI set to 1.0", attr_name)
        else:
            di = round(min_rate / max_rate, 4)

        metrics['disparate_impact'][attr_name] = di
        logger.info("DI [%s]: %.4f", attr_name, di)

        # Demographic Parity Difference
        parity_diff = round(max_rate - min_rate, 4)
        metrics['demographic_parity_diff'][attr_name] = parity_diff
        logger.info("Parity diff [%s]: %.4f", attr_name, parity_diff)

        # 4/5ths severity tracking
        if di < 0.8:
            raw_sev = (0.8 - di) * 100
            per_attr_severity.append((raw_sev, attr_name))
            bias_detected = True
            logger.info("BIAS DETECTED [%s] DI=%.4f raw_sev=%.2f", attr_name, di, raw_sev)

        # Plain-English explanation
        explanation_text = _plain_english_explanation(
            attr_name, rates, di, parity_diff, detection_reason
        )
        bias_explanations.append({
            'attribute': attr_name,
            'group_rates': rates,
            'di_ratio': di,
            'parity_diff': parity_diff,
            'bias_detected': di < 0.8,
            'detection_reason': detection_reason,
            'explanation_text': explanation_text,
        })

    # ── Weighted-max severity aggregation ─────────────────────────────────────
    # PRIMARY: worst attribute -> full magnitude (max ~80)
    # SECONDARY: each additional -> 25% (prevents all-100 saturation)
    # Final cap 85 so ethical_score base-15 gives Low=15, Med=~50, High=90+
    if per_attr_severity:
        per_attr_severity.sort(reverse=True)
        primary_sev    = per_attr_severity[0][0]
        secondary_sev  = sum(s * 0.25 for s, _ in per_attr_severity[1:])
        severity_score = round(min(primary_sev + secondary_sev, 85.0), 2)
    else:
        severity_score = 0.0

    # ── Build detected_attributes summary (for UI display) ────────────────────
    detected_attributes = [
        {
            'col': a['col'],
            'reason': a['reason'],
            'n_unique': a['n_unique'],
            'dtype': a['dtype'],
        }
        for a in sensitive_attrs
    ]

    result = {
        'bias_detected':       bias_detected,
        'severity_score':      severity_score,
        'decision_labels':     {
            'positive': pos_label,
            'negative': neg_label,
            'detected_values': [str(v) for v in all_labels[:10]],
        },
        'detected_attributes': detected_attributes,
        'metrics':             metrics,
        'bias_explanations':   bias_explanations,
    }

    logger.info("analyze_bias complete. bias=%s severity=%.2f attrs=%d",
                bias_detected, severity_score, len(detected_attributes))
    logger.info("=" * 60)
    return result
