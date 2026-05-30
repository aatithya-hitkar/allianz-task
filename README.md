# TeleConnect Customer Churn & Retention Agent

This repository contains the complete solution for the TeleConnect AI/ML Engineer Take-Home Assessment. It is divided into two parts:
1. **Part 1:** Data cleaning, Exploratory Data Analysis (EDA), and training predictive ML models (Logistic Regression & XGBoost).
2. **Part 2:** A production-ready AI Retention Agent using Google ADK and an LLM-as-Judge evaluation framework, deployed via a Streamlit web interface.

## 📂 Project Folder Structure

```
├── data/                       # Datasets
│   ├── test_datafile.csv       # Original raw customer data
│   └── cleaned_datafile.csv    # Processed data after cleaning pipeline
├── src/                        # Core Application Code
│   ├── config.py               # Environment configuration and secrets management
│   ├── data_processing.py      # Data cleaning and quality summary pipeline
│   ├── model_building.py       # ML training, evaluation, and visualizations
│   ├── tools.py                # Functions invoked by the AI agent (mock/ml tools)
│   ├── agent.py                # The Google ADK Agent orchestration logic
│   └── app.py                  # The Streamlit web application
├── evaluation/                 # Evaluation Framework (LLM-as-Judge)
│   ├── test_suite.json         # 12 handcrafted test cases covering various scenarios
│   ├── eval_metrics.py         # Automated metric calculations (Latency, Accuracy)
│   ├── llm_judge.py            # The LLM-as-Judge logic and scoring rubric
│   ├── run_evaluation.py       # Script to execute the test suite and evaluate responses
│   └── scorecard_analysis.md   # Final analysis, deep dives, and CI/CD roadmap
├── models/                     # Saved ML Artifacts
│   ├── churn_model_xgb.pkl     # Trained XGBoost model
│   ├── churn_model_lr.pkl      # Trained Logistic Regression model
│   ├── churn_scaler.pkl        # StandardScaler used during training
│   └── churn_features.pkl      # List of expected model features
├── mock_data/                  # Data for Agent Tools
│   ├── customers.json          # 10 diverse mock customer profiles for lookup
│   ├── offers.json             # Retention offers segmented by risk and contract
│   ├── interactions_log.jsonl  # Simulated interaction database
│   └── escalations_log.jsonl   # Simulated escalation database
├── notebooks/                  # Notebooks
│   └── Part1_Churn_Analysis.ipynb # Jupyter notebook with Part 1 narrative
├── requirements.txt            # Python dependencies (CPU only)
└── README.md                   # This file
```

## ⚙️ How to Setup

This project uses Python 3.11+. The dependencies are optimized for CPU environments to prevent bulky unnecessary downloads (e.g., CUDA libraries).

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory (or update the provided one) and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 🚀 How to Run the Application

The main deliverable for Part 2 is the interactive retention agent built with Streamlit. To run the web UI:

```bash
streamlit run src/app.py
```
This will launch a local web server (usually at `http://localhost:8501`). The UI features a chat interface where you can speak to the agent. Below every message, the UI explicitly renders the **tool orchestration trace** to provide transparency into how the agent retrieved and processed information.

## 📊 How to Evaluate using LLM-as-Judge

We implemented a robust evaluation framework that runs 12 predefined test cases against the agent. A separate LLM evaluates the agent's performance across 4 dimensions: Factual Correctness, Tool Use Appropriateness, Actionability, and Hallucination.

To run the evaluation pipeline:

```bash
python evaluation/run_evaluation.py
```

Running this script will:
1. Iterate over all 12 test cases in `evaluation/test_suite.json`.
2. Intercept the trace to calculate `Tool Selection Accuracy` and `Parameter Extraction Accuracy`.
3. Invoke the `llm_judge.py` to grade the output on a 1-5 scale using strict anchors.
4. Output the results to `evaluation/evaluation_results.json` and `evaluation/evaluation_scorecard.csv`.
5. For a complete analysis of the failures, successes, and reasoning, refer to `evaluation/scorecard_analysis.md`.

## 🔮 Future Updates & CI/CD Roadmap

To integrate this evaluation pipeline into an automated CI/CD pipeline (e.g., GitHub Actions):
1. **Containerize the Evaluator**: Package `run_evaluation.py` into a lightweight Docker image.
2. **Mock the API layer**: In a real CI pipeline, the `predict_churn` tool should be mocked to return static responses so the evaluation isolates *LLM routing logic* without wasting compute on XGBoost inference.
3. **Threshold Blocking**: Configure the CI step to automatically fail the build if `Tool Selection Accuracy` drops below 95% or if `Avg Hallucination` drops below 4.5. 
4. **Parallelization**: Switch `run_agent()` to the asynchronous ADK API (`runner.run_async()`) and use `asyncio.gather` to evaluate the 12+ test cases in parallel, drastically reducing CI pipeline wait times.
