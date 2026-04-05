"""
ethical_score.py — Ethical Risk Score & Business Impact Engine
===============================================================
Takes bias metrics from bias_detection.py and produces:
  - risk_score (0–100)
  - risk_level (Low / Medium / High)
  - explanation (narrative summary)
  - recommendations (actionable improvement steps)
  - business_impact (context-aware risk bullets)
  - feature_importance (heuristic weights)
"""

import logging

logger = logging.getLogger(__name__)

# ── Business Impact Library ───────────────────────────────────────────────────

_BUSINESS_IMPACT = {
    'hiring': [
        "⚠️ **Legal risk**: Discriminatory hiring patterns may violate employment equality laws (e.g. Title VII, Equality Act).",
        "📉 **Diversity loss**: Biased selection reduces workforce diversity, limiting creativity and innovation.",
        "💸 **Talent gaps**: Excluding qualified candidates reduces the talent pool and raises recruitment costs.",
        "🗞️ **Reputational damage**: Publicised bias can harm employer brand, impacting future talent attraction.",
        "⚖️ **Lawsuit exposure**: Affected candidates may file discrimination claims, leading to costly legal proceedings.",
    ],
    'loan': [
        "⚠️ **Regulatory penalties**: Biased lending decisions may violate fair lending laws (e.g. Equal Credit Opportunity Act).",
        "📉 **Customer trust loss**: Discriminatory credit decisions erode customer confidence and loyalty.",
        "⚖️ **Legal liability**: Affected applicants can challenge decisions, exposing the institution to lawsuits.",
        "💸 **Market risk**: Refusing qualified borrowers reduces revenue and market share.",
        "🗞️ **Reputational damage**: Media coverage of lending bias can significantly damage brand perception.",
    ],
    'medical': [
        "⚠️ **Patient safety risk**: Biased diagnostic or treatment decisions can directly harm patient health outcomes.",
        "⚖️ **Medical malpractice**: Discriminatory care decisions may constitute medical negligence or malpractice.",
        "📉 **Health equity impact**: Systemic bias worsens health disparities across demographic groups.",
        "🗞️ **Regulatory scrutiny**: Healthcare regulators may investigate bias in automated clinical decision tools.",
        "💸 **Financial penalties**: Regulatory fines and loss of accreditation can have severe financial consequences.",
    ],
    'general': [
        "⚠️ **Compliance risk**: Biased automated decisions may violate anti-discrimination legislation.",
        "📉 **Operational impact**: Unfair outcomes reduce stakeholder trust and long-term organisational effectiveness.",
        "⚖️ **Legal exposure**: Affected parties may challenge discriminatory decisions through legal action.",
        "🗞️ **Reputational risk**: Public discovery of algorithmic bias can cause significant brand damage.",
        "💸 **Financial consequences**: Legal costs, fines, and lost business from bias often exceed the cost of prevention.",
    ],
}

_CONTEXT_KEYWORDS = {
    'hiring': ['hire', 'hiring', 'job', 'employ', 'recruit', 'candidate', 'applicant', 'position', 'career'],
    'loan': ['loan', 'credit', 'mortgage', 'finance', 'lending', 'borrow', 'debt', 'bank'],
    'medical': ['medical', 'health', 'patient', 'clinical', 'diagnostic', 'treatment', 'hospital', 'care'],
}


def _infer_context(dataset_name: str) -> str:
    """Infer business context from dataset filename."""
    name_lower = (dataset_name or '').lower()
    for context, keywords in _CONTEXT_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            logger.info("Business context inferred: '%s' from name '%s'", context, dataset_name)
            return context
    logger.info("No specific context matched for '%s' — using 'general'", dataset_name)
    return 'general'


def _build_recommendations(biased_attrs: list, di_metrics: dict) -> list:
    """Build a rich, attribute-specific recommendation list."""
    recs = []

    if biased_attrs:
        recs.append("🔍 **Audit your training data** — identify and remove historical patterns that reflect past discrimination.")
        recs.append("⚖️ **Apply fairness-aware algorithms** — use techniques like re-weighting, adversarial debiasing, or post-processing threshold adjustment.")
        recs.append("📊 **Improve dataset diversity** — ensure all demographic groups are adequately represented in training data.")

        if 'Gender' in biased_attrs or any('gender' in a.lower() for a in biased_attrs):
            recs.append("🚫 **Remove gender as a direct input feature** unless legally justified for the decision context.")
            recs.append("🔬 **Audit data collection processes** for gender-related sampling bias in historical records.")

        if 'Age' in biased_attrs or any('age' in a.lower() for a in biased_attrs):
            recs.append("📋 **Review age-related criteria** — ensure age-based thresholds are legally and ethically justified.")
            recs.append("🏛️ **Check compliance** with age discrimination laws (e.g. ADEA in the US, Equality Act in the UK).")

        for attr, di in di_metrics.items():
            if attr not in ('Gender', 'Age') and di < 0.8:
                recs.append(f"🔎 **Investigate '{attr}' column** — it shows significant disparate impact (DI={di:.2f}) that requires domain-specific review.")

        recs.append("🔁 **Implement continuous monitoring** — run fairness audits on every model retrain and after new data ingestion.")
        recs.append("📝 **Document model decisions** — maintain an audit trail to demonstrate fairness compliance to regulators.")
    else:
        recs.append("✅ **Continue monitoring** — fairness should be re-evaluated as new training data arrives.")
        recs.append("📈 **Expand coverage** — consider checking additional sensitive attributes not yet analysed.")
        recs.append("🔁 **Periodic re-auditing** — schedule regular audits as the decision context or dataset evolves.")
        recs.append("📝 **Document this audit** — record the fairness metrics as a baseline for future comparison.")
        recs.append("🧪 **Stress-test with edge cases** — validate model behaviour on underrepresented subgroups.")

    return recs


# ── Main Function ─────────────────────────────────────────────────────────────

def generate_ethical_score(bias_metrics: dict, dataset_name: str = '') -> dict:
    """
    Calculate Ethical Risk Score and all associated outputs.

    Parameters
    ----------
    bias_metrics : dict   Output from analyze_bias()
    dataset_name : str    Used for business context inference

    Returns
    -------
    dict with keys:
        risk_score, risk_level, explanation, recommendations,
        business_impact, feature_importance, context
    """
    logger.info("generate_ethical_score called. dataset_name='%s'", dataset_name)

    severity = bias_metrics.get('severity_score', 0.0)
    bias_detected = bias_metrics.get('bias_detected', False)
    logger.debug("Input severity_score=%.2f  bias_detected=%s", severity, bias_detected)

    # ── Risk Score ─────────────────────────────────────────────────────────────
    # Base of 15 ensures even a "fair" dataset shows baseline model uncertainty.
    # Severity contribution is linear with the DI violation magnitude.
    risk_score = min(15.0 + severity, 100.0)

    if risk_score <= 30:
        risk_level = "Low Risk"
    elif risk_score <= 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    logger.info("risk_score=%.1f  risk_level=%s", risk_score, risk_level)
    logger.debug("Ethical risk score generated: %.1f -> %s", round(risk_score, 1), risk_level)

    # ── Biased Attributes ──────────────────────────────────────────────────────
    metrics = bias_metrics.get('metrics', {})
    di_metrics = metrics.get('disparate_impact', {})
    biased_attrs = [attr for attr, di in di_metrics.items() if di < 0.8]
    logger.debug("Biased attributes (DI < 0.8): %s", biased_attrs)

    # ── Explanation ────────────────────────────────────────────────────────────
    if biased_attrs:
        attrs_str = ', '.join(f'**{a}**' for a in biased_attrs)
        explanation = (
            f"The AI decision process shows statistically significant bias in: {attrs_str}. "
            f"The selection rate for the disadvantaged group falls below 80% of the advantaged group, "
            f"violating the 4/5ths fairness rule. Immediate investigation and remediation is recommended."
        )
    else:
        explanation = (
            "The AI decision process appears relatively fair across all analysed demographic attributes "
            "based on the 4/5ths rule. No significant disparate impact was detected. "
            "Continue monitoring as data evolves."
        )

    # ── Recommendations ────────────────────────────────────────────────────────
    recommendations = _build_recommendations(biased_attrs, di_metrics)

    # ── Business Impact ────────────────────────────────────────────────────────
    context = _infer_context(dataset_name)
    business_impact = _BUSINESS_IMPACT.get(context, _BUSINESS_IMPACT['general'])
    if not bias_detected:
        business_impact = [
            "✅ No significant bias detected — current risk of discrimination-related legal exposure is low.",
            "📋 Maintain documentation of this audit result for compliance purposes.",
            "🔁 Schedule regular re-audits to ensure sustained fairness as the model evolves.",
        ]

    # ── Feature Importance (heuristic) ─────────────────────────────────────────
    feat_importance = {}
    bias_exps = bias_metrics.get('bias_explanations', [])
    if bias_exps:
        # Weight biased attributes higher
        total_attrs = len(bias_exps)
        for exp in bias_exps:
            attr = exp['attribute']
            if exp['bias_detected']:
                feat_importance[attr] = round(0.4 + (0.2 / max(total_attrs, 1)), 3)
            else:
                feat_importance[attr] = round(0.15, 3)
        # Add Experience if not already present
        if 'Experience' not in feat_importance:
            remaining = max(1.0 - sum(feat_importance.values()), 0.1)
            feat_importance['Experience'] = round(remaining, 3)
    else:
        feat_importance = {'Experience': 0.6, 'Gender': 0.2, 'Age': 0.2}

    logger.debug("Feature importance: %s", feat_importance)
    logger.info("generate_ethical_score complete.")

    return {
        'risk_score': round(risk_score, 1),
        'risk_level': risk_level,
        'explanation': explanation,
        'recommendations': recommendations,
        'business_impact': business_impact,
        'feature_importance': feat_importance,
        'context': context,
    }
