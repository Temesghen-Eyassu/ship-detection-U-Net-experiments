SAR Ship Segmentation Using U-Net: A Comparative Study of Data Augmentation and Dual-Head Learning
Abstract
This project investigates a deep learning approach for ship segmentation in Sentinel-1 SAR imagery. The task is challenging due to speckle noise, class imbalance, and the small, elongated shape of ships. A U-Net model is used as a baseline and evaluated under different data augmentation strategies. In addition, a dual-head U-Net is introduced in the final experiment to predict both segmentation masks and skeleton representations. Six experiments are conducted, starting from a baseline model, followed by different augmentation settings, and finally dual-head architecture. The results show that data augmentation improves robustness and training behavior but has limitations with fluctuating metrics. In contrast, the dual head model significantly improves segmentation quality by capturing structural information and the resulting metrics show improved generalizations. Overall, the study demonstrates that architectural improvements are more effective than data augmentation alone for SAR ship segmentation. 
1. Pipeline Overview
The pipeline implemented in this project follows a standard deep learning workflow for semantic segmentation using PyTorch. The process begins with Sentinel-1 SAR imagery, where dual-polarization channels (VV and VH) are normalized and divided into smaller patches to enable efficient model training.
Depending on the experimental setup, these patches are further processed using progressively refined data augmentation strategies across Experiments 2–5, including geometric and radiometric transformations. In the final experiment, additional feature engineering is introduced through the computation of distance maps and skeleton representations, which provide structural information about ship shapes.
The prepared data is loaded using PyTorch DataLoaders and passed to the segmentation models. Two main modeling approaches are considered in this study: a baseline U-Net (Experiment 1) and its augmented variants (Experiments 2–5), followed by an extended dual-head U-Net (Experiment 6) that jointly predicts segmentation masks and structural skeletons.
Training is performed using the Adam optimizer with Binary Cross-Entropy (BCE) loss for segmentation. In the dual-head setup, an additional skeleton prediction loss is incorporated and combined with the segmentation loss using a weighted formulation to encourage structural learning.
Model performance is evaluated using the Dice coefficient and Intersection over Union (IoU). Training behavior is further analyzed by monitoring the progression of loss, Dice, and IoU values across epochs for both training and validation sets to assess the convergence and generalization.
The overall deep learning pipeline used in this project can be summarized as follows:
                                 
Figure 1: Deep Learning Pipeline Framework used across all experiments
As shown in figure 1, the overall deep learning pipeline starts from Sentinel-1 SAR image data, which undergoes preprocessing and configuration of dataset parameters. The data is then organized into training and validation sets, where data augmentation is applied to improve generalization. These augmentations include geometric transformations such as flipping and rotation, as well as radiometric transformations such as intensity variations. PyTorch DataLoader is used to efficiently feed data into the deep learning model. The model is trained using a defined loss function within an iterative training loop. Performance is evaluated using Dice coefficient and Intersection over Union (IoU), and the final stage involves generating and visualizing prediction results.
This pipeline is consistently applied in Experiments 1–5, where different augmentation strategies are systematically evaluated using a baseline U-Net architecture. In Experiment 6, the framework is extended to a dual-head U-Net that jointly teaches segmentation masks and skeleton representations through an additional structural supervision branch.
2. Data Source and Preparation
The dataset used in this study is derived from Sentinel-1 Synthetic Aperture Radar (SAR) imagery processed using Google Earth Engine for the coastal region of Shanghai, China, covering the period from January to August 2025. The data was filtered to ensure consistency by selecting Interferometric Wide Swath (IW) mode, ascending orbit direction, and dual polarization (VV and VH).
To reduce speckle noise and temporal variability, a median composite was generated across the selected time range. This approach helps preserve stable backscatter characteristics while suppressing noise. The final dataset includes speckle filtered VV and VH polarization channels.
Ground truth annotations were created by manually digitizing approximately 208 ships using QGIS. These annotations were converted into binary segmentation masks, where ships are labeled as foreground and all other pixels as background. The SAR images and corresponding masks were then divided into 64 × 64-pixel patches to facilitate efficient training of the U-Net model.
 
Figure 2: SAR image over Shanghai waters. Left: VV polarization channel. Right: VH polarization channel.
 
Figure 3: Ground truth mask for the entire SAR scene, created in QGIS. Pixels are labeled as 0 (background) and 1 (ships). This mask represents all ship locations across the coastal area and serves as the reference for extracting training and validation patches (e.g., 64×64 pixel patches). It illustrates the spatial distribution of ships and provides a complete overview of the annotated area used in this study
The following figure shows the patches used in SAR Ship Detection like VV, VH, Ground Truth, and Predicted Mask 
 
Figure 4: Example patch (64×64 pixels) from the SAR dataset. 
3. Problem Description
The objective of this work is to perform binary semantic segmentation of ships in Sentinel-1 SAR imagery at the pixel level. This task is particularly challenging due to the presence of strong imbalance, as ships occupy only a small portion of the image. Additionally, SAR imagery is inherently affected by speckle noise, which reduces image clarity and complicates feature extraction. Ships often appear as thin, elongated, and irregular structures, making them difficult to distinguish from some backgrounds like small islands. The goal of this study is therefore to improve both segmentation accuracy and structural consistency of the predicted masks.
4. Baseline Model
The baseline model used in this study is a standard U-Net architecture consisting of an encoder–decoder structure with skip connections. These skip connections help preserve spatial information across different scales, which is important for semantic segmentation tasks.
The model performs binary segmentation using a single output head, predicting ship versus background. In all baseline experiments, the model is trained using the Adam optimizer with Binary Cross-Entropy (BCE) loss. Dice coefficient and Intersection over Union (IoU) are used as evaluation metrics to measure segmentation performance.
In the initial experiment, a basic level of data augmentation is applied, including geometric transformations such as horizontal flipping and rotation, as well as radiometric transformations such as brightness and contrast adjustments. These augmentations introduce limited variability while preserving the structural integrity of the SAR imagery.
This setup serves as the reference point for subsequent experiments, where augmentation strategies are progressively modified. Structural information in the form of skeleton prediction is introduced only in the final experiment using a dual-head architecture.
5. Motivation for Improvement
Initial observations from the baseline experiment reveal several limitations. Although the model can learn meaningful representations, the validation performance shows noticeable fluctuations, particularly in the Dice and IoU curves. These fluctuations indicate that the model does not generalize consistently across epochs.
The training curves are relatively smooth, while the validation metrics vary more significantly. This gap between training and validation behavior suggests that the model is sensitive to variations in the data and lacks robustness when exposed to unseen samples.
Furthermore, the instability observed in the validation metrics implies that the model struggles to maintain consistent predictions, which often results in fragmented segmentation outputs, especially for thin and elongated ship structures.
To address these issues, two improvement strategies are explored: the use of data augmentation to enhance robustness and the introduction of dual-head architecture to improve structural awareness.
6. Experimental Design and Graph-Based Analysis
To systematically investigate the impact of data augmentation and architectural modifications, a total of six experiments were conducted. These experiments are designed in a progressive manner, where each setup is built upon the observations of the previous one. This allows for a controlled and interpretable comparison of how different augmentation strategies and model designs influence training behavior, stability, and segmentation performance.

The experiments are summarized as follows:
Experiment	Description	Key Modification
Experiment 1	Baseline U-Net	Basic augmentation
Experiment 2	Strong augmentation	Increased augmentation intensity
Experiment 3	Very strong augmentation	Aggressive transformations
Experiment 4	Balanced augmentation	Moderate augmentation
Experiment 5	Light augmentation	Minimal transformations
Experiment 6	Dual-head U-Net	Structural learning via skeleton prediction
Table 1: Overview of the Experiments
6.1 Loss Curve Analysis
Across all experiments, the training loss exhibits a consistent pattern: a rapid decrease during the initial epochs followed by gradual stabilization which is after ~10 to ~20 epochs, all curves flatten. This behavior indicates that all models are capable of learning and optimizing effectively, regardless of augmentation strategy or architectural design.
However, differences emerge when examining the validation loss:
	In Experiment 1 (Baseline), validation loss follows a relatively smooth trajectory with minor fluctuations, suggesting stable but limited generalization. 
	In Experiments 2 and 3 (Strong and Very Strong Augmentation), validation loss in both experiments shows slightly variation than baseline but remains largely stable. 
	In Experiments 4 and 5 (Balanced and Light Augmentation), validation loss stabilizes again, indicating that reducing augmentation strength leads to improved training consistency. 
	In Experiment 6 (Dual-Head U-Net), all loss components (segmentation and skeleton losses) converge smoothly, demonstrating that the additional supervision does not negatively affect optimization but instead contributes to stable learning dynamics. 
Overall, while loss curves provide useful insight into optimization behavior, they appear relatively similar across experiments and therefore do not fully capture differences in segmentation quality.


6.2 Dice and IoU Metric Analysis
More pronounced differences between experiments are observed in the Dice coefficient and Intersection over Union (IoU), which provide a more direct measure of segmentation performance.
Experiment 1: Baseline
The baseline model achieves moderate Dice and IoU values. While the general trend is increasing, the validation curves show frequent fluctuations and occasional sharp drops. This indicates that the model is able to learn meaningful representations but lacks robustness when applied to unseen data.
Experiment 2: Strong Augmentation
With stronger augmentation, the model is exposed to a more diverse dataset. However, this comes at the cost of stability:
	Dice and IoU curves show large oscillations 
	Frequent spikes and drops are observed 
	Performance is inconsistent across epochs 
This suggests that although augmentation increases variability, it also introduces noise that makes optimization more difficult.
Experiment 3: Very Strong Augmentation
In this experiment, augmentation is further intensified. The impact on performance is clearly negative:
	Dice and IoU remain close to zero during early training 
	The model struggles to learn meaningful features initially 
	Even after partial recovery, metrics remain highly unstable 
This confirms that excessive augmentation disrupts feature learning and significantly degrades model performance.
Experiment 4: Balanced Augmentation
Reducing augmentation intensity leads to noticeable improvements:
	Dice and IoU curves become smoother as compared with the previous experiments
	Fewer abrupt drops are observed 
	Validation metrics improve more consistently, which is still noisy but less extreme
This indicates that a moderate level of augmentation achieves a better balance between variability and stability.
Experiment 5: Light Augmentation
With minimal augmentation, the model achieves its most stable behavior among augmentation-based approaches:
	Training and validation curves are closely aligned 
	Dice and IoU increase steadily with minimal fluctuations 
	Convergence is smooth and consistent 
However, despite improved stability, peak performance remains similar to Experiment 4. This suggests that reducing augmentation improves consistency but does not significantly enhance overall segmentation accuracy.
Experiment 6: Dual-Head U-Net
The final experiment introduces dual-head architecture that predicts both segmentation masks and skeleton representations.
This modification leads to the most significant improvement:
	Dice and IoU increase rapidly and stabilize at higher values 
	Validation closely follows training, indicating strong generalization 
	Metric fluctuations are minimal compared to all previous experiments 
These results demonstrate that structural supervision enables the model to better capture geometric properties of ships, resulting in improved segmentation quality and continuity.
6.3 Comparative Analysis Across Experiments
When comparing all experiments, several key patterns emerge:
a.	Loss curves alone are insufficient to evaluate segmentation performance
Although loss decreases consistently across all experiments, it does not reflect differences in segmentation quality. 
b.	Data augmentation introduces a trade-off 
Strong augmentation increases variability but reduces stability and on the other hand light augmentation improves stability but offers limited performance gains 
c.	Excessive augmentation is harmful
Very strong augmentation disrupts feature learning and delays convergence. 
d.	Architectural changes have the greatest impact
The dual-head U-Net significantly outperforms all augmentation-based approaches, both in terms of stability and segmentation accuracy. 
These experiments also highlight the role of scaling in deep learning, as increasing data variability (through augmentation) and model complexity (through dual-head architecture) affect both training stability and performance.
6.4 Key Insight
The experiments clearly show that:
While data augmentation improves robustness, it is not sufficient to address the structural challenges of SAR ship segmentation. In contrast, incorporating structural supervision through a dual-head architecture leads to substantial improvements in both segmentation performance and generalization.
It is important to note that slight variations in training and validation metrics were observed across different runs of the same experiment. This behavior is expected in deep learning due to several sources of randomness, including weight initialization, data shuffling, data augmentation, and non-deterministic GPU operations.
Additionally, the relatively small dataset size and the sparsity of ship pixels further amplify metric variability, particularly for Dice and IoU, which are sensitive to small changes in segmentation outputs. Therefore, fluctuations in validation metrics do not necessarily indicate instability in the training process but rather reflect the inherent nature of the learning procedure.
 
Figure 5: Metrics summary for all experiments. 
Figure 5 shows stable and similar loss convergence across all experiments, but clear performance separation in Dice and IoU where dual-head U-Net significantly outperforms all augmentation-based approaches.
7. Results and Discussion
The results of the conducted experiments reveal clear differences between augmentation-based approaches and architectural modifications in terms of both performance and stability.
7.1 Impact of Data Augmentation
Data augmentation was introduced to improve generalization by increasing variability in the training data. However, the experiments demonstrate that its effectiveness is highly dependent on the strength of the applied transformations.
Strong and very strong augmentation (Experiments 2 and 3) introduce significant instability during training. This is reflected in the large oscillations observed in Dice and IoU metrics. While these approaches increase data diversity, they are making it more difficult for the model to learn consistent and meaningful features.
On the other hand, balanced and light augmentation (Experiments 4 and 5) provide a better trade-off between variability and stability as compared to Experiment 2 and 3 but they are not the best option. These configurations result in smoother training behavior and more consistent validation performance. However, despite improved stability, the overall segmentation accuracy does not increase significantly compared to the baseline. This indicates that augmentation alone is not sufficient to address the core challenges of the task.
7.2 Limitations of Augmentation-Based Approaches
The experiments highlight an important limitation: data augmentation primarily improves robustness but does not enhance the model’s ability to capture structural properties of objects.
In SAR ship segmentation, ships appear as thin, elongated, and sometimes fragmented structures. Pixel-wise loss functions combined with augmentation are often insufficient to enforce geometric consistency, leading to discontinuities and fragmented predictions.
This explains why even the most stable augmentation setup (Experiment 5) does not yield substantial improvements in Dice and IoU scores.
7.3 Impact of Dual-Head Architecture
The introduction of the dual-head U-Net (Experiment 6) results in the most significant improvement across all evaluated metrics.
By jointly predicting segmentation masks and skeleton representations, the model benefits from structural supervision. This additional learning objective encourages the network to capture geometric features such as continuity, connectivity, and shape.
As a result:
	Dice and IoU reach the highest values among all experiments 
	Metric curves are smooth and stable 
	Training and validation performance are closely aligned 
These observations indicate improved generalization and reduced overfitting.
7.4 Key Insight
The experiments demonstrate that:
While data augmentation improves robustness, it has limited impact on peak performance.
In contrast, architectural modifications that incorporate structural information provide substantial improvements in segmentation quality.
8. Conclusion
This project presented a systematic investigation of SAR ship segmentation using deep learning, focusing on the effects of data augmentation and architectural improvements within a U-Net framework.
The results show that data augmentation plays an important role in improving model robustness and training stability. However, it introduces distortions that hinder feature learning and destabilize optimization.
More importantly, the study demonstrates that architectural modifications have a significantly greater impact on performance. The proposed dual-head U-Net, which incorporates skeleton-based structural supervision, consistently outperforms all augmentation-based approaches. This model achieves higher segmentation accuracy, improved structural consistency, and more stable generalization.
These findings highlight the importance of integrating structural information into deep learning models, particularly for remote sensing tasks involving thin and geometrically complex objects such as ships in SAR imagery.


Limitations and Future Work
Despite the promising results, this study has some limitations. The dataset is relatively small and geographically limited, which may affect the generalization of the model to other regions or conditions.
Future work could explore:
	Larger and more diverse SAR datasets 
	Alternative structural representations 
	Advanced multi-task learning strategies 
	Scaling experiments using high-performance computing environments
References
1. 	Zhu, X.X., Tuia, D., Mou, L., Xia, G.S., Zhang, L., Xu, F., Fraundorfer, F.: Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources, (2017)
2. 	Ma, L., Liu, Y., Zhang, X., Ye, Y., Yin, G., Johnson, B.A.: Deep learning in remote sensing applications: A meta-analysis and review, (2019)
3. 	Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F.: LNCS 9351 - Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. (2015)
4. 	Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., Liang, J.: UNet++: A Nested U-Net Architecture for Medical Image Segmentation. (2018)
5. 	Yasir, M., Jianhua, W., Mingming, X., Hui, S., Zhe, Z., Shanwei, L., Colak, A.T.I., Hossain, M.S.: Ship detection based on deep learning using SAR imagery: a systematic literature review. Soft comput. 27, 63–84 (2023). https://doi.org/10.1007/s00500-022-07522-w
6. 	Kim, H.C., Lee, H.T., Cho, I.S.: Vessel detection for maritime traffic management using U-Net with backbone networks. Ocean Engineering. 340, (2025). https://doi.org/10.1016/j.oceaneng.2025.121943
7. 	Satheeskumaran, S., Zhang, Y., Valentina, ·, Balas, E., Hong, T.-P., Pelusi, D.: Intelligent Computing for Sustainable Development Communications in Computer and Information Science 2121.
8. 	Shorten, C., Khoshgoftaar, T.M.: A survey on Image Data Augmentation for Deep Learning. J. Big Data. 6, (2019). https://doi.org/10.1186/s40537-019-0197-0
9. 	Wang, J., Perez, L.: The Effectiveness of Data Augmentation in Image Classification using Deep Learning.
10. 	Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P.W., Bauer, U., Menze, B.H.: clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. (2022). https://doi.org/10.1109/CVPR46437.2021.01629
 
