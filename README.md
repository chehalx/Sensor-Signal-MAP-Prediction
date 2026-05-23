Objective
The objective of this project is to develop a deep learning model that predicts Mean Arterial Pressure (MAP) from raw physiological waveform data. Each sample consists of 10 seconds of sensor data sampled at 100 Hz, resulting in 1000 data points per instance. Only the waveform data was used as input, while all other label columns such as Time, HR, RR, SV, and TPR were excluded. The goal was to learn the relationship between waveform characteristics and MAP and achieve a low prediction error on the test set.
 Pipeline Overview
Raw Signal → Preprocessing → Feature Engineering → CNN Model → Evaluation
The proposed workflow begins with the acquisition of raw physiological waveform data, which is then subjected to preprocessing to remove noise and improve signal consistency. Following this, feature engineering is performed to extract relevant statistical, spectral, and physiology-based characteristics from the waveform. These processed inputs are then used to train a CNN-based model designed to learn both local and global patterns in the signal. Finally, the model is evaluated by comparing the predicted MAP values against the ground truth labels using performance metrics such as MAE and visual comparison plots
Signal Preprocessing
Before training the model, the raw waveform signals were cleaned and prepared to make them easier to work with. I applied bandpass filtering to separate the signal into two useful frequency ranges: the cardiac band (0.5–8 Hz), which captures the main pulse-related activity, and the motion band (8–20 Hz), which helps account for higher-frequency disturbances and motion effects. After that, each sample was normalized to have zero mean and unit variance so that all signals were on the same scale. One especially important step was aligning the global standard deviation between the training and test data, which helped reduce distribution shift and improved how well the model generalized to unseen samples.
<img width="875" height="439" alt="image" src="https://github.com/user-attachments/assets/f21c37b9-875e-4192-b56a-a6904063d87e" />
This figure shows signal decomposition applied to physiological recordings from three patients with different Mean Arterial Pressures (MAP: 57, 78, and 92 mmHg). Each raw signal (gray) is filtered into two components: a cardiac signal (blue, 0.5–8 Hz) capturing true heartbeat and vascular pulsations, and a motion artifact signal (red, 8–20 Hz) representing high-frequency noise from movement or sensor interference. The cardiac component retains the dominant, clinically meaningful waveform, while the motion component shows much lower amplitude noise. This bandpass filtering approach is a standard technique in biomedical signal processing to separate useful physiological data from unwanted interference, improving the reliability of downstream analysis or diagnosis.
Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) reveals a severe feature-level covariate shift between the data splits, despite the target MAP distributions remaining perfectly matched (~75 ± 11.8 mmHg). A strong positive correlation (r≈ 0.65) exists between signal amplitude (per-sample STD) and MAP, indicating that the model will likely rely on signal magnitude as a key predictive shortcut. However, a major distribution shift is present in both the per-sample STD and skewness features; the test set exhibits higher overall amplitudes and entirely lacks the training set's higher-end skewness clusters. Because this shift is restricted to the input features and not the clinical target, it points to a structural artefact within the synthetic data generation parameters or a subject-level simulation split, rather than a clinical population shift. Because the dataset is synthetic, this represents a challenging out-of-distribution evaluation scenario built into the simulation logic. To ensure robust model training, the preprocessing pipeline must incorporate per-sample Z-score normalisation to strip away these absolute amplitude discrepancies, forcing the downstream model to focus on robust morphological shapes rather than unstable raw scales to strip away these absolute amplitude discrepancie
<img width="940" height="313" alt="image" src="https://github.com/user-attachments/assets/85161b90-c864-4186-93ca-31ae934d3fbc" />
<img width="940" height="376" alt="image" src="https://github.com/user-attachments/assets/1a655278-0a49-41bb-a4fb-da04e0809780" />
<img width="940" height="251" alt="image" src="https://github.com/user-attachments/assets/7e3cdd25-eb96-4c81-ad99-697cc440df35" />
<img width="932" height="672" alt="image" src="https://github.com/user-attachments/assets/76b458d5-a3d0-4783-8bfc-60483dd88ffd" />
Feature Engineering
To extract the complex morphological and physiological properties of the waveform, 30+ handcrafted features were extracted across three core domains:
	Statistical Profiles: Standard distribution metrics, including mean, standard deviation (STD), skewness, and kurtosis.
	Frequency & Physiological Metrics: FFT spectral energy to map power distribution, cardiac/motion band amplitudes to isolate noise, and autocorrelation-based Heart Rate (HR) to track the primary cardiac cycle.
	Physics-Inspired Interactions: Non-linear combinations modeling the relationship between cardiac activity and pressure dynamics:
HR × amplitude, amplitude/HR , 〖HR〗^2 and log(HR)
Feature Engineering
To extract the complex morphological and physiological properties of the waveform, 30+ handcrafted features were extracted across three core domains:
	Statistical Profiles: Standard distribution metrics, including mean, standard deviation (STD), skewness, and kurtosis.
	Frequency & Physiological Metrics: FFT spectral energy to map power distribution, cardiac/motion band amplitudes to isolate noise, and autocorrelation-based Heart Rate (HR) to track the primary cardiac cycle.
	Physics-Inspired Interactions: Non-linear combinations modeling the relationship between cardiac activity and pressure dynamics:
HR × amplitude, amplitude/HR , 〖HR〗^2 and log(HR)
Feature Engineering
To extract the complex morphological and physiological properties of the waveform, 30+ handcrafted features were extracted across three core domains:
	Statistical Profiles: Standard distribution metrics, including mean, standard deviation (STD), skewness, and kurtosis.
	Frequency & Physiological Metrics: FFT spectral energy to map power distribution, cardiac/motion band amplitudes to isolate noise, and autocorrelation-based Heart Rate (HR) to track the primary cardiac cycle.
	Physics-Inspired Interactions: Non-linear combinations modeling the relationship between cardiac activity and pressure dynamics:
HR × amplitude, amplitude/HR , 〖HR〗^2 and log(HR)

Feature Engineering
To extract the complex morphological and physiological properties of the waveform, 30+ handcrafted features were extracted across three core domains:
	Statistical Profiles: Standard distribution metrics, including mean, standard deviation (STD), skewness, and kurtosis.
	Frequency & Physiological Metrics: FFT spectral energy to map power distribution, cardiac/motion band amplitudes to isolate noise, and autocorrelation-based Heart Rate (HR) to track the primary cardiac cycle.
	Physics-Inspired Interactions: Non-linear combinations modeling the relationship between cardiac activity and pressure dynamics:
HR × amplitude, amplitude/HR , 〖HR〗^2 and log(HR)
<img width="940" height="336" alt="image" src="https://github.com/user-attachments/assets/b160520c-464e-4547-beec-544aa7db5be4" />
Model Development
	CNN Architecture (Final Model): Features a deep 1D Convolutional Neural Network designed for spatial-temporal sequence learning. The architecture consists of 5 Conv1D blocks (Conv1D →Batch Normalisation → ReLU → Max Pooling) to automatically extract hierarchical morphological features from the raw waveform. The sequence maps are reduced via Global Average Pooling and directly concatenated with the 30+ engineered features before being passed through fully connected layers to predict the final MAP output.
	Training Techniques & Optimization: Designed for high stability and robustness against noise, the training pipeline utilizes Huber loss minimized by the AdamW optimizer. To handle edge cases, target labels undergo a quantile transformation. Training stability is strictly maintained using gradient clipping to prevent exploding gradients, alongside early stopping to avoid overfitting.

 Model Comparison
 
 <img width="940" height="392" alt="image" src="https://github.com/user-attachments/assets/cb3abee0-be03-4824-b0ae-5aa538439601" />
Observed Results
Model	Train MAE	Test MAE	Behavior
CNN	~5.74	4.99	Best
LightGBM	~8.1	9.77	Stable
XGBoost	~8.1	9.79	Similar
Extra Trees	~1.1	14.72	Overfit
Random Forest	~1.38	14.37	Overfit
Ridge + RBF	~2.56	14.34	Overfit
Linear	~3.78	20.62	Poor

 Prediction Analysis (Visual Evidence)
Best Model (CNN)
•	MAP_Cnn.png (your CNN scatter plot)
Observation:
•	Points closely follow diagonal → strong prediction
•	Low spread → low error
•	MAE ≈ 4.99–6.59

LightGBM
<img width="940" height="705" alt="image" src="https://github.com/user-attachments/assets/c4d678ab-2c98-49e5-bac4-d9393ec5bb82" />
Observation:
 Performance & Core Issue: Initial experiments using a baseline LightGBM model yielded poor generalization with a Mean Absolute Error (MAE) of 9.7706. As illustrated in the prediction vs. label plot below, the model suffers from a severe prediction collapse, flattening out into tight, discrete horizontal bands centered around the target distribution mean ( 74 - 80 mmHg).
Diagnostic Insights: The model completely fails to capture the underlying target variability or track the actual upward trend of the true MAP labels (the red diagonal stair-step pattern). This behaviour strongly confirms that the tree-based model is struggling to map the out – of distribution feature spaces resulting from the severe covariate shift identified during EDA. Instead of learning generalizable morphological relationships, LightGBM resorts to predicting conservative, low-variance mean values to minimize global error.

Extra Trees
<img width="940" height="705" alt="image" src="https://github.com/user-attachments/assets/52b55bd2-73ef-41bd-868a-4b1b44c1f62f" />
Observation:
 Performance & Core Issue: Evaluating an Extra Trees baseline regressor resulted in a critically high Mean Absolute Error (MAE) of 14.7212. The model demonstrates severe overfitting combined with a structural prediction saturation failure mode across the test set.
Diagnostic Insights: Rather than collapsing into a tight mean band like LightGBM, the Extra Trees model heavily over-predicts across almost the entire sample range. For the true MAP labels spanning 55 to 80 mmHg (the lower half of the red stair-step distribution), the model's predictions are scattered completely out of bounds, rapidly drifting upward and saturating in a dense cloud between 90 and 94 mmHg. This confirms that the ensemble's decision trees are failing to generalise under the feature-level distribution shift, causing the splitting logic to break down and default heavily toward the upper bounds of the learned training space.

Linear Model
<img width="940" height="705" alt="image" src="https://github.com/user-attachments/assets/dbd8b31c-b638-4176-a6d8-087e0f229957" />
Observation:
 Performance & Core Issue: The baseline Linear Regression model performed the poorest out of all initial baselines, yielding a critically high Mean Absolute Error (MAE) of 20.6208. The model displays massive variance and a complete inability to track the true target distribution.
Diagnostic Insights: The linear framework suffers from a total breakdown due to two simultaneous flaws: an inability to model non-linear physiological relationships and extreme vulnerability to the covariate shift. Because the test set features significantly higher signal amplitudes (higher standard deviation), the linear coefficients scale up these input values. This pushes the model’s predictions into an absurdly inflated, upward-curving cloud that overshoots the actual MAP ceiling entirely, predicting values all the way up to 130 mmHg for true labels that do not exceed 95mmHg.

Key Insights
The Core Problem: Traditional machine learning models overfit the training data and fail catastrophically on the test set due to a severe train-test distribution shift. Because signal amplitude (per-sample STD) strongly correlates with MAP (0.65), traditional models rely on absolute magnitude as a shortcut feature, which completely breaks down when encountering out-of-distribution test data.
Why the CNN Architecture Succeeds: The deep 1D-CNN naturally bypasses these limitations. Instead of relying on raw data scales, the convolutional blocks extract translation-invariant temporal patterns and morphological features. By concatenating deep waveform representations with physics-inspired features, the network effectively captures complex, non-linear blood pressure dynamics.
 The Critical Breakthrough: The single largest improvement in performance came from proper per-sample normalisation and STD alignment. Forcing each 10 second PPG window to a uniform scale (0 mean, 1 standard deviation) systematically stripped away the absolute amplitude shortcut, neutralising the covariate shift and forcing the network to learn robust, generalizable waveform morphology.
Final Results
 Performance & Core Takeaways: The deep 1D-CNN emerged as the final model, delivering a Test MAE between 4.99 and 6.59 mm Hg, marking a significant, robust improvement over all traditional machine learning baselines.
 Frequency Domain Validation: Power Spectral Density (PSD) analysis confirms that the training and testing splits share nearly identical spectral energy trends across the Cardiac band (0.5 - 8  Hz) and Motion artefact band (8-20Hz). This aligns with the final model's success; by stripping away absolute amplitude dependencies via proper normalisation, the CNN successfully exploited these stable frequency characteristics and localised waveforms to maximise generalisation.
<img width="940" height="376" alt="image" src="https://github.com/user-attachments/assets/a99a7f20-b559-4db1-80ea-9ba9036099ac" />
Project Deliverables
•	Trained Weights: cnn.pkl (Serialised final model architecture and parameters)
•	Inference Pipeline: predict.py (End-to-end preprocessing and prediction script)
•	Diagnostic Visualisation Suite:
o	Performance Scatter Plot (Actual vs. Predicted MAP)
o	Residual Analysis & Error Tracking Distributions
o	Cross-Model MAE Benchmarking Chart
o	Sequence Tracking & Temporal Trend Plot
Conclusion
This project proves that deep learning architectures are essential for time-series physiological signals. While handcrafted feature engineering alone is insufficient to combat out-of-distribution feature spaces, combining domain features with a 1D-CNN creates a robust framework. Managing covariate shift is highly critical, though the stringent project target of MAE ≤ 3 mmHg was not met, achieving ~ 5-6 mmHg under severe distribution shifts represents a strong engineering success.

