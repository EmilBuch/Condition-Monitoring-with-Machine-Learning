# Condition-Monitoring-with-Machine-Learning
This work is from my master thesis: Condition Monitoring with Machine Learning: A Data-Driven Framework for Quantifying Wind Turbine Energy Loss.

Abstract—Wind energy significantly contributes to the global
shift towards renewable energy, yet operational challenges, such
as Leading-Edge Erosion on wind turbine blades, notably reduce
energy output. This study introduces an advanced, scalable
machine learning framework for condition monitoring of wind
turbines, specifically targeting improved detection of anomalies
using Supervisory Control and Data Acquisition data. The
framework effectively isolates normal turbine behavior through
rigorous preprocessing, incorporating domain-specific rules and
anomaly detection filters, including Gaussian Mixture Models
and a predictive power score. The data cleaning and feature
selection process enables identification of deviations indicative of
performance degradation, facilitating estimates of annual energy
production losses. The data preprocessing methods resulted in
significant data reduction, retaining on average 31% of the
original SCADA data per wind farm. Notably, 24 out of 35
turbines exhibited clear performance declines. At the same
time, seven improved, and four showed no significant changes
when employing the power curve feature set, which consisted
of wind speed and ambient temperature. Models such as Ran-
dom Forest, XGBoost, and KNN consistently captured subtle
but persistent declines in turbine performance. The developed
framework provides a novel approach to existing condition
monitoring methodologies by isolating normal operational data
and estimating annual energy loss, which can be a key part
in reducing maintenance expenditures and mitigating economic
impacts from turbine downtime.

<img src="https://github.com/user-attachments/assets/77e91d91-e595-4559-b885-c20199d709bd" alt="Alt Text" style="width:30%; height:auto;">


<img src="https://github.com/user-attachments/assets/fb70ffbf-7b5a-4ddf-b0ac-5ad9469a919b" alt="Alt Text" style="width:40%; height:auto;">


<img src="https://github.com/user-attachments/assets/d020115d-30d1-4b4c-9801-eed91c333142" alt="Alt Text" style="width:40%; height:auto;">

<img src="https://github.com/user-attachments/assets/69ed50f8-e0ff-457a-8a54-4c4c8c28da38" alt="Alt Text" style="width:40%; height:auto;">

