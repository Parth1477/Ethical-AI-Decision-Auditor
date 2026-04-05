# Ethical AI Decision Auditor

## 1. Problem Statement
AI systems are increasingly used to make critical decisions affecting lives (hiring, loans, law enforcement). However, these systems can inherit human biases, leading to unfair outcomes. There is a need for a tool that can audit AI decisions to detect bias, provide explainability, and calculate an ethical risk score.

## 2. Objectives
- Audit AI decisions for fairness and bias.
- Provide transparency in AI decision making through explainability.
- Generate actionable ethical risk scores (0-100).
- Provide recommendations to mitigate identified biases.
- Visualize fairness metrics via an intuitive dashboard.

## 3. Scope of the System
The system accepts structured decision data (CSV or manual input) and analyzes statistical disparities across sensitive attributes like Gender, Age, or Experience. It produces an Ethics Report highlighting disparate impacts, selection rates, feature importance, and overall risk level.

## 4. Functional Requirements
- **Data Input:** Users must be able to upload CSV files and manually enter decision records.
- **Bias Detection:** The system must calculate disparate impact and identify features with selection rates violating the 4/5ths rule (< 0.8 ratio).
- **Ethical Scoring:** Map severity metrics to a normalized 0-100 scale.
- **Reporting:** Display intuitive dashboards mapping risk gauges and attribute selection distributions using Chart.js.

## 5. Non-Functional Requirements
- **Performance:** Dataset analysis should return results within 2 seconds.
- **Usability:** System must be easy to navigate without advanced data science knowledge.
- **Scalability:** The SQLite backend and Flask app should be refactorable into PostgreSQL for enterprise workloads.

## 6. System Architecture
- **Frontend:** HTML5, CSS3 (Custom Design System), JS (Chart.js)
- **Backend:** Flask Framework (Python), Pandas for data manipulation, Scikit-learn logic elements.
- **Data Storage:** SQLite `database.db` with standard relational schemas.

## 7. UML Diagrams Descriptions

- **Use Case Diagram:** Actors (Auditor, Admin) interact with Use Cases: Upload Data, View Dashboard, Generate Report, and Administer System.
- **Class Diagram:** `User`, `Dataset`, `DecisionEntry`, `AuditResult`. Where `Dataset` aggregates multiple `DecisionEntry` models.
- **Sequence Diagram:** User -> UI -> Flask App -> `bias_detection.py` -> `ethical_score.py` -> Database -> UI.
- **Activity Diagram:** Start -> Prompt Data Input -> [Is Valid CSV?] -> Extract Attributes -> Calculate DI -> Compute Risk Score -> Show Dashboard.
- **DFD Level 0:** External Entity (User) sends [Dataset] to the System (Ethical AI Auditor), which outputs [Analysis Report].
- **DFD Level 1:** Sub-processes: 1.0 Ingest Data -> 2.0 Calculate Bias -> 3.0 Generate Risk Profile -> 4.0 Render Dashboard.
- **ER Diagram:** Entity `datasets` (1 to N) `decisions`. Entity `datasets` (1 to 1) `audit_results`.

## 8. Testing Information

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| CSV Upload Validation | Invalid file (.txt) | Error "File format not supported. Upload CSV." |
| Bias DI Calculation | Gender A selection rate 0.5, Gender B 0.9 | DI = 0.55 (Raises Bias Alert) |
| Risk Score Generation | DI=0.55 on Gender, DI=0.9 on Age | Ethical Risk > 31 (Medium Risk) |
| Visualization Render | Valid Dataset JSON | Chart.js canvases render properly |

## 9. Future Enhancements
- Real-time API integration for auditing live production models.
- Support for complex fairness metrics such as Equalized Odds and Predictive Parity.
- Exporting raw DataFrames directly to compliance PDF.
