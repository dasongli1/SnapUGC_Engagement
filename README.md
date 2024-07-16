# SnapUGC_Engagement
Delving deep into Engagement Prediction of Short Videos (ECCV 2024)

Several sample frames of short-form videos dataset:
<center><img src="figures/samples.png "width="70%"></center>

**Abstract**: Understanding and modeling the popularity of User Generated Content (UGC) short videos on social media platforms presents a critical challenge with broad implications for content creators and recommendation systems. This study delves deep into the intricacies of predicting engagement for newly published videos with limited user interactions. Surprisingly, our findings reveal that Mean Opinion Scores from previous video quality assessment datasets do not strongly correlate with video engagement levels.
To address this, we introduce a substantial dataset comprising 130,000 real-world UGC short videos from Snapchat. 
Rather than relying on view count, average watch time, or rate of likes, we propose two metrics: normalized average watch percentage (NAWP) and engagement continuation rate (ECR) to describe the engagement levels of short videos.
Comprehensive multi-modal features, including visual content, background music, and text data, are investigated to enhance engagement prediction. With the proposed dataset and two key metrics, our method demonstrates its ability to predict engagements of short videos purely from video content.

**Keywords**: Engagement Prediction, Short-form Videos

### Key Metrics
we propose two metrics: normalized average watch percentage (NAWP) and engagement continuation rate (ECR). We visualize their distritbuion of short videos of different categories in the following.
<center><img src="figures/distributions.png "width="70%"></center>

### To download the dataset:
Please refer to [this page](https://github.com/dasongli1/SnapUGC_Engagement/blob/main/dataset/readme.md)

### Multi-modal features
<center><img src="figures/multi-modal.png "width="70%"></center>

### Citation
If our work is useful for your research, please consider citing:

```bibtex
@InProceedings{li2024Delving,
    author = {Li, Dasong and Li, Wenjie and Lu, Baili and Li, Hongsheng and Ma, Sizhuo and Krishnan, Gurunandan and Wang, Jian},
    title = {Delving Deep into Engagement Prediction of Short Videos},
    booktitle = {ECCV},
    year = {2024}
}
```
