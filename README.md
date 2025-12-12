# Beauty Product R&D - Strategy Dashboard

This repository contains the source code and data for a dynamic business intelligence dashboard designed to transform unstructured customer feedback into a **financially backed strategic roadmap**.

The primary goal is to shift customer experience initiatives from reactive cost centers to proactive revenue drivers by quantifying the financial impact of every customer issue.

## ✨ Key Business Insights

The dashboard focuses on two core questions answered by the **Priority Score** and **Value-to-Cost Ratio (ROI)**. You can filter all findings by Action Type or Net Impact using the sidebar.

| Metric | Key Finding (Overall) | Value |
| :--- | :--- | :--- |
| **Total Net Impact** | The maximum untapped financial opportunity. | **$99.83 Million** |
| **Global ROI** | The aggregate return on investment. | **1.62:1** |
| **Most Efficient Topic** | High return with minimal spend. | **ROI up to 11.75:1** |

---

## 🛠️ Technology Stack

* **Language:** Python
* **Data Processing:** Pandas
* **Dashboarding:** Streamlit (For interactive web application)
* **Version Control:** Git & Git LFS (Used for handling large data files like `phase4_final_dashboard_data.csv`)

## ⚙️ How to Run the Dashboard Locally

Follow these steps to set up and run the interactive dashboard on your local machine.

### 1. Prerequisites & Cloning

Ensure you have Python (3.8+), Git, and Git LFS installed.

```bash
git clone [YOUR_REPOSITORY_URL_HERE]
cd [YOUR_REPOSITORY_NAME]
```

Set up a virtual environment (recommended) - this will save disk space
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run Streamlit App to see functioning dashboard frontend
```bash
streamlit run app.py
```
