Results and Performance Analysis
Table of Contents
* Overall Performance(#overall-performance)
* Per-Class Analysis(#per-class-analysis)
* Confusion Matrix(#confusion-matrix)
* ROC Curves(#roc-curves)
* Latent Space Visualization(#latent-space-visualization)
* Reconstruction Quality(#reconstruction-quality)
* Comparison with Baselines(#comparison-with-baselines)

Overall Performance
Test Set Metrics
Evaluated on MIT-BIH Arrhythmia Database test set (21,892 samples):
---------------------------------------------
| |Metric | |Value |
---------------------------------------------
| |Overall Accuracy | |96.4% |
---------------------------------------------
| |Macro F1-Score | |0.912 |
---------------------------------------------
| |Weighted F1-Score | |0.964 |
---------------------------------------------
| |Macro Precision | |0.908 |
---------------------------------------------
| |Macro Recall | |0.915 |
---------------------------------------------
| |Inference Time (CPU) | |8.3ms per sample |
---------------------------------------------
| |Inference Time (GPU) | |0.9ms per sample |


Per-Class Analysis
Detailed Performance Breakdown
---------------------------------------------
| |Class | |Support | |Precision | |Recall | |F1-Score | |Accuracy |
---------------------------------------------
| |Normal (N) | |18,118 | |0.994 | |0.997 | |0.995 | |99.2% |
---------------------------------------------
| |Supraventricular (S) | |556 | |0.889 | |0.924 | |0.906 | |94.1% |
---------------------------------------------
| |Ventricular (V) | |1,448 | |0.981 | |0.965 | |0.973 | |97.3% |
---------------------------------------------
| |Fusion (F) | |162 | |0.857 | |0.895 | |0.875 | |89.5% |
---------------------------------------------
| |Unknown (Q) | |1,608 | |0.920 | |0.905 | |0.912 | |92.7% |

Key Observations
✅ Strengths:
* Excellent performance on Normal beats (99.2%)
* Strong Ventricular detection (97.3%)
* Handles class imbalance well
⚠️ Challenges:
* Fusion beats are hardest to classify (89.5%)
	* Only 162 test samples (rare class)
	* Morphologically similar to multiple classes
* Supraventricular beats show occasional confusion with Normal

Confusion Matrix
Normalized Confusion Matrix
Predicted →     N      S      V      F      Q
True ↓
Normal (N)   [0.997  0.001  0.001  0.000  0.001]
Supra  (S)   [0.050  0.924  0.007  0.000  0.019]
Ventr  (V)   [0.010  0.002  0.965  0.012  0.011]
Fusion (F)   [0.037  0.000  0.068  0.895  0.000]
Unknwn (Q)   [0.056  0.012  0.015  0.000  0.905]

Common Misclassifications
1. 
2. S → N (5.0%)
	1. Supraventricular beats occasionally misclassified as Normal
	2. Both have similar QRS morphology in some cases
3. 
4. F → V (6.8%) and F → N (3.7%)
	1. Fusion beats are combinations of ventricular and normal
	2. Small sample size makes learning difficult
5. 
6. Q → N (5.6%)
	1. Some unknown beats resemble normal morphology
	2. Heterogeneous class by definition

ROC Curves
Area Under ROC Curve (AUC)
---------------------------------------------
| |Class | |AUC | |Interpretation |
---------------------------------------------
| |Normal (N) | |0.999 | |Excellent discrimination |
---------------------------------------------
| |Supraventricular (S) | |0.982 | |Excellent |
---------------------------------------------
| |Ventricular (V) | |0.995 | |Excellent |
---------------------------------------------
| |Fusion (F) | |0.971 | |Excellent despite small samples |
---------------------------------------------
| |Unknown (Q) | |0.988 | |Excellent |

Macro Average AUC: 0.987
All classes show excellent discriminative ability (AUC > 0.97), indicating the model learns robust decision boundaries.

Latent Space Visualization
t-SNE Analysis
When visualizing the 128D latent space in 2D using t-SNE:
Observations:
* ✅ Clear clustering by arrhythmia type
* ✅ Normal beats form tight, well-separated cluster
* ✅ Ventricular beats cluster distinctly
* ⚠️ Supraventricular shows some overlap with Normal
* ⚠️ Fusion samples scattered (small sample size)
Cluster Purity:
---------------------------------------------
| |Class | |Cluster Purity | |Nearest Neighbor Accuracy |
---------------------------------------------
| |Normal | |98.5% | |99.1% |
---------------------------------------------
| |Supraventricular | |87.3% | |91.2% |
---------------------------------------------
| |Ventricular | |95.8% | |96.9% |
---------------------------------------------
| |Fusion | |79.2% | |85.1% |
---------------------------------------------
| |Unknown | |89.1% | |91.8% |


Reconstruction Quality
Reconstruction Error by Class
---------------------------------------------
| |Class | |Mean MSE | |Std MSE | |Median MSE |
---------------------------------------------
| |Normal | |0.0012 | |0.0008 | |0.0010 |
---------------------------------------------
| |Supraventricular | |0.0019 | |0.0014 | |0.0015 |
---------------------------------------------
| |Ventricular | |0.0015 | |0.0011 | |0.0012 |
---------------------------------------------
| |Fusion | |0.0028 | |0.0021 | |0.0023 |
---------------------------------------------
| |Unknown | |0.0021 | |0.0016 | |0.0017 |

Visual Quality
Normal Beats:
* Excellent reconstruction of P wave, QRS complex, and T wave
* Preserves fine details in morphology
Abnormal Beats:
* Captures distinctive features (wide QRS in ventricular, etc.)
* Slight smoothing in high-frequency components
