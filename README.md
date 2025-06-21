# Condition-Monitoring-with-Machine-Learning
This work is from my master thesis: Condition Monitoring with Machine Learning: A Data-Driven Framework for Quantifying Wind Turbine Energy Loss.

#### Paper available at: https://arxiv.org/abs/2506.13012

## Abstract
Wind energy significantly contributes to the global
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


## The Problem
The end goal of this research is to quantify the potential energy loss resulting from performance degradation over time. To asses such a state, it is necessary to establish a healthy model state, representative of the turbines expected output. However, identifying such a state is a non-trivial task, even for a single turbine, if event logs are not provided as part of the data, and even sometimes with provided logs. An extensive framework to identify said periods of healthy model states and clean the data of anomalies is, therefore, a necessary step for further research.

An example of anomalous behavior can be observed in the following figure of the power curve:

<img src="https://github.com/user-attachments/assets/d020115d-30d1-4b4c-9801-eed91c333142" alt="Alt Text" style="width:50%; height:auto;">

The extensive framework implemented to accommodate the data cleaning and identifying healthy states can be observed:

<img src="https://github.com/user-attachments/assets/77e91d91-e595-4559-b885-c20199d709bd" alt="Alt Text" style="width:40%; height:auto;">

The Predictive Power Score (PPS) is implemented to work in a temporal setting and utilized as a direct measure of the predictive quality of the data on the power before and after NB-filters. The PPS gives a direct insight into the effectiveness of the filtering applied to the data and allows for the selection of healthy model states. Such an example can be observed on:
<img src="https://github.com/user-attachments/assets/fb70ffbf-7b5a-4ddf-b0ac-5ad9469a919b" alt="Alt Text" style="width:50%; height:auto;">

where the PPS is not only improved across the operating history of the turbine, but further history can potentially be utilized in the following quantification of energy loss assessment.

<img src="https://github.com/user-attachments/assets/69ed50f8-e0ff-457a-8a54-4c4c8c28da38" alt="Alt Text" style="width:50%; height:auto;">



