import os
import uuid
import sqlite3
import logging
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from utils.bias_detection import analyze_bias
from utils.ethical_score import generate_ethical_score

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── App Config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# SECURITY FIX 1: Load SECRET_KEY from environment variable.
# Never hardcode secrets in source code. Fall back to a random key for dev only.
_fallback_key = os.urandom(32).hex()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', _fallback_key)
if 'SECRET_KEY' not in os.environ:
    logger.warning(
        "WARNING: SECRET_KEY not set in environment variables. "
        "Using a random ephemeral key — sessions will not persist across restarts. "
        "Set SECRET_KEY env var before deploying to production."
    )

# SECURITY FIX 2: Hard file-size limit (5 MB). Prevents DDOS via large uploads.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = 'database.db'
MINIMUM_REQUIRED_COLUMN = 'decision'
MIN_ROWS = 4


# ── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('File too large. Maximum allowed size is 5 MB.', 'danger')
    logger.warning("Upload rejected: file exceeded 5 MB limit.")
    return redirect(url_for('upload'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error.")
    flash('An internal server error occurred. Please try again.', 'danger')
    return redirect(url_for('index'))


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_dataframe(df):
    """
    Validate an uploaded dataframe. Returns (ok: bool, error_msg: str).
    """
    if df.empty or len(df) < MIN_ROWS:
        return False, f"Dataset too small — need at least {MIN_ROWS} rows, found {len(df)}."

    col_names_lower = [c.strip().lower() for c in df.columns]
    if MINIMUM_REQUIRED_COLUMN not in col_names_lower:
        return False, (
            "Dataset must contain a 'Decision' column with binary outcomes "
            "(e.g. Selected/Rejected, Approved/Denied, Accepted/Declined). "
            "If your decision column has a different name, rename it to 'Decision'."
        )

    # Check for at least 2 unique decision values
    dec_idx = col_names_lower.index(MINIMUM_REQUIRED_COLUMN)
    dec_col = df.columns[dec_idx]
    unique_vals = df[dec_col].dropna().unique()
    if len(unique_vals) < 2:
        label = unique_vals[0] if len(unique_vals) else 'none'
        return False, (
            f"Decision column only contains one value ('{label}'). "
            "At least two distinct outcome labels are required for bias analysis."
        )

    return True, ""


def _safe_delete_file(filepath):
    """Delete a file silently — used to clean up PII-containing uploads."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("Cleaned up uploaded file: %s", filepath)
    except OSError as e:
        logger.warning("Could not delete file %s: %s", filepath, e)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    logger.info("Homepage loaded.")
    return render_template('index.html')


@app.route('/upload')
def upload():
    logger.info("Upload page loaded.")
    return render_template('upload.html')


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    filepath = None   # track for cleanup

    if 'dataset' not in request.files:
        flash('No file part found in the request.', 'danger')
        logger.warning("Upload: no file part in request.")
        return redirect(url_for('upload'))

    file = request.files['dataset']

    if file.filename == '':
        flash('No file selected. Please choose a CSV file.', 'danger')
        logger.warning("Upload: empty filename.")
        return redirect(url_for('upload'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Only CSV files (.csv) are accepted.', 'danger')
        logger.warning("Upload rejected — invalid extension: %s", file.filename)
        return redirect(url_for('upload'))

    # SECURITY FIX 3: Use UUID filename instead of user-supplied name.
    # Prevents path traversal and enumeration of uploaded PII files.
    original_name = secure_filename(file.filename)
    safe_filename  = f"{uuid.uuid4().hex}.csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)
    logger.info("Dataset saved: %s (original: %s)", safe_filename, original_name)

    try:
        # ── Load & Validate ────────────────────────────────────────────────────
        df = pd.read_csv(filepath)
        logger.info("CSV parsed. Columns: %s | Rows: %d", list(df.columns), len(df))

        ok, err_msg = _validate_dataframe(df)
        if not ok:
            flash(err_msg, 'danger')
            logger.error("Validation failed: %s", err_msg)
            return redirect(url_for('upload'))

        # ── Log dataset to DB ──────────────────────────────────────────────────
        conn = get_db_connection()
        conn.execute('INSERT INTO datasets (dataset_name) VALUES (?)', (original_name,))
        conn.commit()
        conn.close()

        # ── Bias Analysis ──────────────────────────────────────────────────────
        logger.info("Starting bias analysis...")
        bias_results = analyze_bias(filepath)
        logger.info("Bias analysis done. severity=%.2f detected=%s attrs=%s",
                    bias_results['severity_score'], bias_results['bias_detected'],
                    [a['col'] for a in bias_results.get('detected_attributes', [])])

        # ── Ethical Score ──────────────────────────────────────────────────────
        logger.info("Generating ethical score...")
        ethical_results = generate_ethical_score(bias_results, dataset_name=original_name)
        logger.info("Ethical score: %.1f (%s)", ethical_results['risk_score'], ethical_results['risk_level'])

        # ── Save to DB ─────────────────────────────────────────────────────────
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO audit_results (bias_score, ethical_risk, explanation, recommendation) VALUES (?,?,?,?)',
            (
                bias_results['severity_score'],
                ethical_results['risk_score'],
                ethical_results['explanation'],
                json.dumps({
                    'recommendations':      ethical_results['recommendations'],
                    'business_impact':      ethical_results['business_impact'],
                    'feature_importance':   ethical_results['feature_importance'],
                    'metrics':              bias_results['metrics'],
                    'bias_explanations':    bias_results['bias_explanations'],
                    'detected_attributes':  bias_results.get('detected_attributes', []),
                    'decision_labels':      bias_results['decision_labels'],
                    'risk_level':           ethical_results['risk_level'],
                    'bias_detected':        bias_results['bias_detected'],
                    'dataset_name':         original_name,
                    'context':              ethical_results['context'],
                })
            )
        )
        audit_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info("Audit saved to DB. audit_id=%d", audit_id)

        return redirect(url_for('dashboard', audit_id=audit_id))

    except pd.errors.EmptyDataError:
        flash('The uploaded CSV file is empty.', 'danger')
        logger.error("Empty CSV uploaded.")
    except pd.errors.ParserError as e:
        flash('Could not parse the CSV file. Please check the format.', 'danger')
        logger.error("CSV parse error: %s", e)
    except json.JSONDecodeError as e:
        flash('Internal error serializing results.', 'danger')
        logger.error("JSON encode error: %s", e)
    except Exception as e:
        flash(f'An unexpected error occurred during analysis: {type(e).__name__}', 'danger')
        logger.exception("Unexpected error in upload_csv.")
    finally:
        # SECURITY FIX 4: Always delete uploaded file after analysis.
        # Do NOT keep PII-containing CSVs on disk indefinitely.
        if filepath:
            _safe_delete_file(filepath)

    return redirect(url_for('upload'))


@app.route('/dashboard')
def dashboard():
    audit_id            = request.args.get('audit_id', type=int)
    risk_score          = 0
    metrics             = {}
    feature_importance  = {}
    bias_explanations   = []
    business_impact     = []
    decision_labels     = {}
    detected_attributes = []
    context             = 'general'
    dataset_name        = ''

    if audit_id:
        try:
            conn = get_db_connection()
            row = conn.execute('SELECT * FROM audit_results WHERE id = ?', (audit_id,)).fetchone()
            conn.close()

            if row:
                risk_score = row['ethical_risk']
                extra = json.loads(row['recommendation'])
                metrics             = extra.get('metrics', {})
                feature_importance  = extra.get('feature_importance', {})
                bias_explanations   = extra.get('bias_explanations', [])
                business_impact     = extra.get('business_impact', [])
                decision_labels     = extra.get('decision_labels', {})
                detected_attributes = extra.get('detected_attributes', [])
                context             = extra.get('context', 'general')
                dataset_name        = extra.get('dataset_name', '')
                logger.info("Dashboard loaded. audit_id=%s risk=%.1f", audit_id, risk_score)
            else:
                flash('Audit record not found.', 'warning')
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON for audit_id=%s", audit_id)
            flash('Audit data appears corrupted or is from an older version.', 'warning')
        except Exception as e:
            logger.exception("Error loading dashboard for audit_id=%s", audit_id)
            flash('Could not load audit data.', 'danger')

    return render_template('dashboard.html',
                           audit_id=audit_id,
                           risk_score=risk_score,
                           metrics=metrics,
                           feature_importance=feature_importance,
                           bias_explanations=bias_explanations,
                           business_impact=business_impact,
                           decision_labels=decision_labels,
                           detected_attributes=detected_attributes,
                           context=context,
                           dataset_name=dataset_name)


@app.route('/results')
def results():
    audit_id            = request.args.get('audit_id', type=int)
    dataset_name        = 'Unknown Dataset'
    risk_level          = 'Unknown'
    risk_score          = 0
    bias_detected       = False
    explanation         = 'No explanation available.'
    recommendations     = []
    business_impact     = []
    bias_explanations   = []
    detected_attributes = []
    context             = 'general'

    if audit_id:
        try:
            conn = get_db_connection()
            row = conn.execute('SELECT * FROM audit_results WHERE id = ?', (audit_id,)).fetchone()
            conn.close()

            if row:
                risk_score  = row['ethical_risk']
                explanation = row['explanation']
                extra = json.loads(row['recommendation'])
                recommendations     = extra.get('recommendations', [])
                business_impact     = extra.get('business_impact', [])
                bias_explanations   = extra.get('bias_explanations', [])
                detected_attributes = extra.get('detected_attributes', [])
                risk_level          = extra.get('risk_level', 'Unknown')
                bias_detected       = extra.get('bias_detected', False)
                dataset_name        = extra.get('dataset_name', 'Uploaded Dataset')
                context             = extra.get('context', 'general')
                logger.info("Results loaded. audit_id=%s risk=%s", audit_id, risk_level)
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON for results audit_id=%s", audit_id)
        except Exception as e:
            logger.exception("Error loading results for audit_id=%s", audit_id)

    return render_template('results.html',
                           audit_id=audit_id,
                           dataset_name=dataset_name,
                           risk_level=risk_level,
                           bias_detected=bias_detected,
                           risk_score=risk_score,
                           explanation=explanation,
                           recommendations=recommendations,
                           business_impact=business_impact,
                           bias_explanations=bias_explanations,
                           detected_attributes=detected_attributes,
                           context=context)


if __name__ == '__main__':
    logger.info("Starting Ethical AI Decision Auditor...")
    app.run(debug=True, port=5000)
