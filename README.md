# SaaS Customer Churn Predictor & Business Explainer

**[Try the Live App on Hugging Face]([https://huggingface.co/spaces/Mofidi80/saas-churn-predictor])**

Predicts customer churn with **97.8% ROC-AUC** and provides plain-English, actionable explanations for retention teams using SHAP. Built to bridge predictive modeling with real-world business decision-making.

---
![Alt text]()
---

## Business Problem & Key Insights
SaaS companies lose 5-7% of annual revenue to silent churn. This project identifies at-risk customers *before* they leave and explains **why**, enabling targeted retention strategies.

** Key Findings from Exploratory Analysis:**
- Customers on **Month-to-Month contracts** are **2.5x more likely to churn** than annual subscribers.
- Churned customers submit an average of **5 support tickets** vs. 2 for retained customers.
- Inactive users average **40 days since last login**, compared to 10 for retained users.
- Low satisfaction scores (<4/10) strongly correlate with contract cancellation.

---

## Technical Stack
| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Modeling** | XGBoost, SHAP, Joblib |
| **Deployment** | Streamlit, Hugging Face Spaces |
| **Evaluation** | ROC-AUC, Precision/Recall, Confusion Matrix |

---

## 🚀 Project Pipeline
1. **Data Ingestion & EDA**: Analyzed 15K+ customer records to identify behavioral and demographic churn drivers.
2. **Feature Engineering**: Created business-aligned metrics (`engagement_score`, `support_intensity`, `contract_risk`).
3. **Model Training**: Trained XGBoost classifier with `scale_pos_weight` to handle class imbalance.
4. **Explainability**: Used SHAP to translate model outputs into actionable retention recommendations.
5. **Deployment**: Wrapped pipeline in a Streamlit app and deployed live on Hugging Face Spaces.

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.978 |
| **Recall (Churn)** | 0.90 |
| **Precision (Churn)** | 0.88 |
| **F1-Score (Churn)** | 0.89 |

> *Why these metrics?* In churn prediction, **Recall** matters most: catching 90% of at-risk customers allows retention teams to intervene before revenue is lost, while high precision ensures marketing budget isn't wasted on loyal users.

---

## How It Works (Explainability)
Instead of a "black box" prediction, the app uses **SHAP values** to map model outputs back to real customer data:
CHURN RISK ALERT: Customer #847
Predicted Churn Probability: 88.2%
TOP 3 DRIVERS:
• Last Login Days Ago: 45 ➔ This strongly INCREASES churn risk.
• Support Tickets: 6 ➔ This strongly INCREASES churn risk.
• Contract Type: Month-to-Month ➔ This strongly INCREASES churn risk.

---

## 🛠️ Run Locally
+ Clone the repository
```console
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```
+ Install dependencies
```console
pip install -r requirements.txt
```

+ Launch the app
```console
streamlit run app.py
```

---

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## Contact
Mohammad Mofidi
* Email: mohammad.mofidi.k@gmail.com
* Linkedin: https://www.linkedin.com/in/mohammad-mofidikhajeh
