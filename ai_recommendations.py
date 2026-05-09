import requests
import os

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"   # free, powerful model



def call_groq_api(prompt, api_key, max_tokens=600):
    """
    Sends a prompt to Groq's free API and returns the AI response.

    Parameters:
        prompt     (str) : your question / data to send to AI
        api_key    (str) : your Groq API key (from .env or UI input)
        max_tokens (int) : max length of AI response (default 600)

    Returns:
        response_text (str) : AI's answer
        OR raises Exception with readable error message
    """

    # Validate key
    if not api_key or len(api_key.strip()) < 10:
        raise Exception(
            "API key is missing or too short.\n"
            "Get a FREE key at: https://console.groq.com\n"
            "Sign up with Google → API Keys → Create API Key"
        )

    # Build HTTP request headers
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type" : "application/json"
    }

    # Build request body (payload)
    payload = {
        "model"   : GROQ_MODEL,
        "messages": [
            {
                "role"   : "system",
                "content": (
                    "You are a professional data analyst and business advisor. "
                    "Give specific, clear, practical advice based on the data provided. "
                    "Always reference the actual numbers and column names from the data. "
                    "Use simple English. Format your response clearly with numbered points."
                )
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        "max_tokens" : max_tokens,
        "temperature": 0.7   # 0=factual, 1=creative — 0.7 is balanced
    }

    # Send the HTTP request
    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30        # wait maximum 30 seconds
        )
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Check your internet connection.")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot reach Groq server. Check your internet.")

    # Handle API errors
    if response.status_code == 401:
        raise Exception(
            "Invalid API key (Error 401).\n"
            "Check your key at: https://console.groq.com"
        )
    if response.status_code == 429:
        raise Exception(
            "Rate limit hit (Error 429).\n"
            "Free tier: 30 requests/minute. Wait 60 seconds and try again."
        )
    if response.status_code != 200:
        raise Exception(
            f"Groq API error {response.status_code}: {response.text[:200]}"
        )

    # Extract text from JSON response
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ------------------------------------------------------------
# FUNCTION 2 — build_drop_prompt
# Converts drop analysis results into a readable text block
# This text is sent to the AI so it understands the data
# ------------------------------------------------------------
def build_drop_prompt(drops_df, stats_dict, value_col, dataset_name=""):
    """
    Builds a text description of detected drops for the AI.

    Parameters:
        drops_df    (DataFrame) : rows with detected drops
        stats_dict  (dict)      : basic statistics
        value_col   (str)       : name of the numeric column
        dataset_name(str)       : name of the uploaded file

    Returns:
        prompt_text (str) : formatted text describing the drops
    """
    lines = []
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Column analysed: {value_col}")
    lines.append("")
    lines.append("SUMMARY STATISTICS:")
    lines.append(f"  Total  : {stats_dict.get('total', 'N/A')}")
    lines.append(f"  Mean   : {stats_dict.get('mean', 'N/A')}")
    lines.append(f"  Max    : {stats_dict.get('max', 'N/A')}")
    lines.append(f"  Min    : {stats_dict.get('min', 'N/A')}")
    lines.append(f"  Rows   : {stats_dict.get('count', 'N/A')}")
    lines.append("")

    if drops_df is None or len(drops_df) == 0:
        lines.append("No significant drops detected in this data.")
    else:
        lines.append(f"DROPS DETECTED: {len(drops_df)} drop(s) found")
        lines.append("-" * 45)
        for i, (_, row) in enumerate(drops_df.iterrows(), 1):
            lines.append(f"Drop #{i}:")
            lines.append(f"  Current value  : {row[value_col]}")
            lines.append(f"  Previous value : {row['prev_value']}")
            lines.append(f"  Change         : {row['change_pct']}%")
            lines.append(f"  Amount dropped : {row['drop_amount']}")
            lines.append(f"  Severity       : {row['severity']}")
            # Add any other columns for context
            for col in drops_df.columns:
                if col not in [value_col, 'prev_value',
                               'change_pct', 'drop_amount', 'severity']:
                    lines.append(f"  {col}: {row[col]}")
            lines.append("")

    return "\n".join(lines)



def build_prediction_prompt(pred_result, value_col):
    """
    Builds a text description of prediction results for the AI.

    Parameters:
        pred_result (dict) : output from prediction.run_full_prediction()
        value_col   (str)  : name of the predicted column

    Returns:
        prompt_text (str)  : formatted text describing predictions
    """
    lines = []
    lines.append(f"PREDICTION RESULTS for column: {value_col}")
    lines.append(f"  Model used    : Linear Regression")
    lines.append(f"  Slope (trend) : {pred_result.get('slope', 'N/A')}")
    lines.append(f"  Direction     : {pred_result.get('direction', 'N/A')}")
    lines.append(f"  MAE (error)   : {pred_result['metrics'].get('MAE', 'N/A')}")
    lines.append(f"  R² score      : {pred_result['metrics'].get('R2', 'N/A')}")
    lines.append(f"  Model quality : {pred_result['metrics'].get('R2_label', 'N/A')}")
    lines.append("")
    lines.append("FUTURE PREDICTIONS:")

    for label, pred in zip(
        pred_result.get('future_labels', []),
        pred_result.get('future_preds', [])
    ):
        lines.append(f"  {label} → {pred:,.2f}")

    return "\n".join(lines)



def get_why_drops(drops_df, stats_dict, value_col,
                  api_key, dataset_name=""):
    """
    Sends drop data to Groq AI and asks WHY the drops happened.

    Returns:
        answer (str) : AI's explanation of why drops occurred
    """
    data_text = build_drop_prompt(
        drops_df, stats_dict, value_col, dataset_name
    )

    prompt = f"""
A data analyst has uploaded a CSV file and detected drops in the data.
Here is the analysis:

{data_text}

Based on this data:
1. Explain in 3-4 sentences WHY these drops most likely happened.
2. List 3 possible root causes for the drops.
3. Mention which drop is the most critical and why.

Be specific — reference the actual numbers and values from the data above.
"""
    return call_groq_api(prompt, api_key, max_tokens=500)



def get_recommendations(drops_df, stats_dict, pred_result,
                        value_col, api_key, dataset_name=""):
    """
    Sends drop data + predictions to AI and asks for recommendations.

    Returns:
        answer (str) : AI's numbered list of recommendations
    """
    drop_text = build_drop_prompt(
        drops_df, stats_dict, value_col, dataset_name
    )
    pred_text = build_prediction_prompt(pred_result, value_col)

    prompt = f"""
A business analyst has uploaded a dataset and needs recommendations.

{drop_text}

{pred_text}

Based on this actual data, give EXACTLY 3 numbered smart recommendations
to fix the drops and improve future performance.

Format exactly like this:
1. [Title of recommendation]: [2-3 specific sentences with actionable steps]
2. [Title of recommendation]: [2-3 specific sentences with actionable steps]
3. [Title of recommendation]: [2-3 specific sentences with actionable steps]

Reference the actual numbers and column names. Be practical and specific.
"""
    return call_groq_api(prompt, api_key, max_tokens=650)



def get_insight_summary(drops_df, stats_dict, pred_result,
                        value_col, api_key, dataset_name=""):
    """
    Generates a complete insight paragraph — the final output.
    Corresponds to the "Insight Generated" box in your diagram.

    Returns:
        answer (str) : complete insight in paragraph form
    """
    drop_text = build_drop_prompt(
        drops_df, stats_dict, value_col, dataset_name
    )
    pred_text = build_prediction_prompt(pred_result, value_col)

    prompt = f"""
Complete data analysis summary:

{drop_text}

{pred_text}

Write a complete INSIGHT REPORT in this format:

INSIGHT SUMMARY:
[2-3 sentences describing what the data shows overall]

KEY FINDING:
[The single most important finding from the data]

TREND:
[Is the trend going up or down? What does this mean?]

IMMEDIATE ACTION REQUIRED:
[The most urgent thing to do based on this analysis]

Reference actual numbers from the data. Keep each section to 1-2 sentences.
Total response should be under 200 words.
"""
    return call_groq_api(prompt, api_key, max_tokens=400)


# ------------------------------------------------------------
# FUNCTION 7 — test_api_key
# Checks if the API key is valid before running analysis
# ------------------------------------------------------------
def test_api_key(api_key):
    """
    Sends a tiny test request to verify the API key works.

    Returns:
        (True, None)           → key is valid
        (False, error_message) → key is invalid
    """
    try:
        result = call_groq_api(
            "Say only the word: VALID",
            api_key,
            max_tokens=5
        )
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
#  SELF TEST — run: python ai_recommendations.py
# ============================================================
if __name__ == '__main__':
    import pandas as pd

    print("=" * 55)
    print("  ai_recommendations.py — self test")
    print("=" * 55)

    # Ask user to enter API key for testing
    api_key = input("\nEnter your Groq API key (or press Enter to skip): ").strip()

    if not api_key:
        print("\n  Skipping API test — no key entered.")
        print("  Get a FREE key at: https://console.groq.com")
    else:
        print("\n[1] Testing API key...")
        ok, err = test_api_key(api_key)
        if ok:
            print("  ✅ API key is valid and working!")
        else:
            print(f"  ❌ {err}")
            exit(1)

       