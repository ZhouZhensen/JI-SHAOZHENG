import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('RF.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "Age", "Non_Hispanic_Black", "PIR", "Below_high_school", "BMI", "SIRI", "Mexican_American"
]

# Streamlit user interface
st.title("Sarcopenia Risk Calculator for Older Adults with Hypertension")

# Age: numerical input
Age = st.number_input("Age:", min_value=18, max_value=100, value=60)

Non_Hispanic_Black = st.selectbox("Non_Hispanic_Black:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

Mexican_American = st.selectbox("Mexican_American:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

PIR = st.number_input("PIR:", min_value=0.00, max_value=6.00, value=3.00, step=0.10)

Below_high_school = st.selectbox("Below_high_school:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

BMI = st.number_input("BMI:", min_value=10.00, max_value=50.00, value=22.00, step=0.10)

SIRI = st.number_input("SIRI:", min_value=0.01, max_value=20.00, value=0.60, step=0.10)

# Process inputs and make predictions
feature_values = [Age, Non_Hispanic_Black, PIR, Below_high_school, Mexican_American, BMI, SIRI]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model assessment, you are at high risk of sarcopenia. "
            f"The model predicts that your probability of having sarcopenia risk is {probability:.1f}%. "
            "We recommend consulting with healthcare professionals for a comprehensive evaluation and considering interventions such as nutritional support and resistance training."
        )
    else:
        advice = (
            f"According to our model assessment, you are at low risk of sarcopenia. "
            f"The model predicts that your probability of not having sarcopenia is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle with balanced nutrition and moderate exercise is important. Please continue regular health check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # 在创建LIME解释器之前，预处理数据
    import numpy as np
    import pandas as pd
    from scipy import stats

    # 复制数据避免修改原始数据
    X_test_clean = X_test.copy()

    # 处理数值型特征
    for col in X_test_clean.select_dtypes(include=[np.number]).columns:
        # 检查标准差是否为0
        if X_test_clean[col].std() == 0:
            # 添加微小噪声
            noise = np.random.normal(0, 1e-6, len(X_test_clean))
            X_test_clean[col] = X_test_clean[col] + noise
            st.info(f"特征'{col}'的标准差为0，已添加微小噪声")

    # 确保没有NaN
    X_test_clean = X_test_clean.fillna(X_test_clean.mean())

    # 创建LIME解释器
    lime_explainer = LimeTabularExplainer(
        training_data=X_test_clean.values,
        feature_names=X_test_clean.columns.tolist(),
        class_names=['健康', '患病'],
        mode='classification',
        discretize_continuous=True,  # 可以保持为True
        kernel_width=3,
        random_state=42
    )

    # 处理输入特征
    features_clean = features.flatten().copy()
    for i in range(len(features_clean)):
        if np.isnan(features_clean[i]):
            features_clean[i] = np.nanmean(features_clean)

    lime_exp = lime_explainer.explain_instance(
        data_row=features_clean,
        predict_fn=model.predict_proba,
        num_samples=1000
    )

