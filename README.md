# Reproduction and Extension of "Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation"

This repository is setup to provide access to the code and data which was used in writing the paper [Reproducing and extending “Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation”](http://parl.ai). The original paper [Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation](https://arxiv.org/abs/1911.03842) utilizes [ParlAI](https://parl.ai/) for training, evaluation, and text generation. The model used in these papers is an 88M parameter transformer with 16 attention heads, 8 encoder layers, 8 decoder layers, and embedding size of 512. This model is pretrained on the [large Reddit](https://pushshift.io/) dataset with approximately 1.7 billion comments. In all experiments conducted in this paper, this pretrained model was fine-tuned on the [LIGHT](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) dataset with some form of data augmentation or training to mitigate the gender bias in the conversations in the game. In large corpora, there is potential for text to contain gender bias, which the models can learn from and generate gender biased text. The goal of these papers is to mitigate gender bias present in corpora and prevent models from generating gender biased text. In order to accomplish this task, the original paper created three bias mitigation techniques.
-	**Counterfactual Data Augmentation**\
This augmentation method uses a set of gendered words and replaces them with their opposite. For example, it swaps king for queen or priest for priestess.
-	**Positively-Biased Data Collection**\
This method utilizes positively biased, crowd-sourced data to shift the gender bias to a more neutral level. Although the amount of data infusion is about 10% of the original dataset, it has a huge effect on shifting the bias present in the dataset.
-	**Bias Controlled Training**\
This method puts each label into a bucket based on the type of bias present in the text. For example, if the text includes male gendered words but no female gendered words, it is assigned to the f0 m1 bucket, but if both female and male gendered words are present, it assigns it to the f1 m1 bucket. The bucket labels are added to the end or training features for the model to learn from when generating a response.
-	**Neutral Generated Data**\
This method is an extension to the original paper. For this method, we originally train a model using counterfactual data augmentation and bias controlled training, and use it to generate responses for the entire training data. Next, we go through the generated text and 90% of the time pick the generated responses that are neutral, such as those belonging to the f0 m0 bucket or f1 m1 bucket with a relatively equal number of male and female gendered words. We use the neutral generated responses to reconstruct the conversations and then train a new model on these neutral conversations.


## Original Paper's Github Repository 

In the original paper, Dinan et al. use the ParlAI infrastructure to run the experiments. The [genderation bias github repository](https://github.com/facebookresearch/ParlAI/tree/master/projects/genderation_bias) gives the necessary information on the paper. The general [Facebook research ParlAI bias github repository](https://github.com/facebookresearch/ParlAI) gives information about the pretrained model and datsets used for various research projects, including the [LIGHT](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) dataset.

## Reproduction and Extension Code

The entire [source code](https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/src/gender_bias_project.ipynb) is documented and sectioned for ease of access in an iPython notebook. Each section in the notebook includes a particular experiment's code and results, which are described in the paper. In addition, extension code and results are provided in the iPython notebook. The code is organized in such a way that to run the notebook for the very first time, you need to run the initial setup section and then the experiment code, starting with the general training subsection. However, all subsequent times, the regular setup can be used, followed by the desired experiment cell. This code is best suited for running on Google Drive with Google Colaboratory with access to a [PyTorch](https://pytorch.org/) compatible GPU, but with minor modifications it can be run on a regular or other virtual machines. More description on how to use the code is provided on the iPython notebook provided in the src folder.

## Dependencies
The dependencies for this project are all met when running the iPython notebook on Google Drive with Google Colaboratory with access to a [PyTorch](https://pytorch.org/) compatible GPU, but if this is not the machine you are using, you need the following dependencies to be met:
-   [PyTorch](https://pytorch.org/) with access to a compatible GPU
-   [ParlAI](https://parl.ai/)
-   [subword-nmt](https://github.com/rsennrich/subword-nmt)
-   [Numpy](https://numpy.org/)
-   [NLTK](https://www.nltk.org/)\
Additionally, general [python 3.7.10](https://www.python.org/downloads/release/python-3710/) libraries need to be installed, such as re, copy, os, json, pickle, random, sys.

## Data Download Instructions
To download the data, please install [ParlAI](https://parl.ai/) using:
```bash
pip install parlai
```
Once ParlAI is installed on the machine, use the command below from the [paper's ParlAI page](https://parl.ai/projects/genderation_bias/) to download all the data from the original paper.
```bash
parlai display_data -t light_genderation_bias
```
This data is also available in the data folder of this repository for ease of access.

## Pretrained Model Download
The pretrained model is automatically downloaded when using the code provided in the iPython notebook in the src folder, but the following command will also download the pretrained model trained on the Reddit dataset used in this paper.
```bash
parlai interactive -mf zoo:tutorial_transformer_generator/model 
```
## General Code and Commands
All the code and commands for data preprocessing, training, evaluation, and paper extensions are provided in the iPython notebook in the src folder. In addition, [ParlAI's documentation](https://parl.ai/docs/index.html) is quite helpful for commands not used in the notebook.  

## Results
Below are the results from reproducing the experiments done by Dinan et al. and for extensions to these experiments. The experiment extensions are both aimed at addressing the high time and monetary cost of positively biased data collection, which requires crowdsourcing data. The first of these extensions is fine-tuning the pretrained Reddit model on the data generated from counterfactual data augmentation and using bias controlled training to determine the impact of excluding positively biased data collection. The second extension still fine-tunes the model using counterfactual data augmentation and bias controlled training, but also includes neutral data we generate (the process for generating this data is described in the "Neutral Generated Data" section above). The results below give the percent gendered words (number of gendered words out of the total number of words in the generated responses), percent male bias (number of male gendered words out of the gendered words), and F1 score for each model for four bins: F0M0, F+M0, F0M+, and F+M+. The test data is split into bins based on the presence of gendered words in the label (the next response in the conversation). F0M0 means there are no gendered words in the label. F+M0 means there is at least one female gendered word but no male gendered words in the label. F0M+ means there are no female gendered words but at least one male gendered word in the label. F+M+ means there is at least one female and one male gendered word in the label. Discussion of the results is included in our [paper](https://arxiv.org/abs/1911.03842).

Figure 1 below shows how each bias mitigation technique is used to mitigate gender bias in the generated text. The plots separate the data into buckets used for bias controlled training to show how these techniques mitigate bias in the generated text. These plots also show how well bias controlled training gives control to the model when generating text by telling the model what type of data it must generate via passing the bucket as part of the features in the episode.  

<p align="center">
  <img src="https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/images/ReproducibilityChartResults.PNG"><br>
  <strong>Figure 1:</strong> Results for Reproducing the Experiments in Original Paper by <a href="https://arxiv.org/abs/1911.03842">Dinan et al.</a><br>
  <br> 
</p>

Figure 2 below shows the results from the original paper and the results from our extensions to the original paper. The two extensions are using counterfactual data augmentation and bias controlled training techniques without the positive-biased data augmentation, and counterfactual data augmentation and bias controlled training when adding our neutral, generated data for data augmentation. The results suggest that adding neutral generated utterances instead of the crowd sourced positively-biased data collection can yield similar or better results than using the "All" method (combining all 3 bias original bias mitigation techniques) from the original paper, and approximately the same or slightly higher F1 scores.  

<p align="center">
  <img src="https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/images/CdaAndBiasAndNeutralDataResults.PNG"><br>
  <strong>Figure 2:</strong> Results for Combining all 3 Bias Mitigation Techniques vs. Counterfactual Data Augmentation and Bias Controlled Training both with and without Neutral Generated Data.</a><br>
  <br> 
</p>

In addition, using neutral generated utterances with counterfactual data augmentation and bias controlled training techniques result in producing more gender-neutral generated text, but maintains similar control on the level of bias in the generated text, as was evident from the similar F1 scores it achieved in Figure 2 for each bucket.

<p align="center">
  <img src="https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/images/GeneratedResponsesPerBinChart.PNG"><br>
  <strong>Figure 3:</strong> Percent of Generated Responses from each Model in each Bin.</a><br>
  <br>
</p>

The tables below show the results from our [paper](https://arxiv.org/abs/1911.03842) for each model, reproducing and extending the orginal paper by [Dinan et al.](https://arxiv.org/abs/1911.03842)
  
  <br>
  
**Results for Each Model for F0M0 Bin:**

|                 Model              | % Gendered Words |   % Male Bias  |    F1 Score    |% Generated Responses in This Bin|
|                  :-:               |        :-:       |      :-:       |       :-:      |              :-:                |
|                Baseline            |        5.48      |      45.14     |      13.22     |             35.11               |
|                  CDA               |        5.35      |	     38.05     |      12.98     |             38.96               |
|                Pos Data            |        5.94      |	     46.50     |      13.06     |             36.31               |
|           Bias Ctrl Training       |        0.69	    |      56.85     |      13.59     |             41.30	              |
|                  All               |        0.32      |      43.53     |      13.75     |             39.41               |
|            CDA + Bias Ctrl         |        0.80      |      44.96     |      14.62     |             41.94               |
|  CDA + Bias Ctrl + Our Gen. Data   |        0.72      |      49.68     |      14.62     |             41.40               |

  <br>

**Results for Each Model for F+M0 Bin:**

|                 Model              | % Gendered Words |   % Male Bias  |    F1 Score    |% Generated Responses in This Bin |
|                  :-:               |        :-:       |      :-:       |       :-:      |              :-:                 |
|                Baseline            |        6.40      |      42.07     |      14.84     |             29.88                | 
|                  CDA               |        6.16      |	     33.85     |      14.27     |	            31.04                |
|                Pos Data            |        7.62      |	     40.88     |      14.99     |	            31.48                |
|           Bias Ctrl Training       |        8.76	    |      4.70      |      15.4      |             34.26                |
|                  All               |        8.25      |      1.95      |      15.92     |             35.02                |
|            CDA + Bias Ctrl         |        7.62      |      4.08      |      15.48     |             33.74                |
|  CDA + Bias Ctrl + Our Gen. Data   |        8.44      |      5.90      |      15.4      |             33.41                |

  <br>

**Results for Each Model for F0M+ Bin:**

|                 Model              | % Gendered Words |   % Male Bias  |    F1 Score    |% Generated Responses in This Bin|
|                  :-:               |        :-:       |      :-:       |       :-:      |              :-:                |
|                Baseline            |        6.90      |      52.35     |      15.12     |             20.38               |
|                  CDA               |        6.46      |	     41.53     |      14.9      |             18.67               |
|                Pos Data            |        7.51      |	     53.53     |      15.41     |             19.92               |
|           Bias Ctrl Training       |        7.36	    |      94.37     |      15.4      |             14.82               |
|                  All               |        7.89      |      97.13     |      17.31     |             13.41               |
|            CDA + Bias Ctrl         |        6.97      |      95.52     |      16.37     |             14.00               |
|  CDA + Bias Ctrl + Our Gen. Data   |        6.55      |      93.41     |      16.6      |             12.98               |

  <br>

**Results for Each Model for F+M+ Bin:**

|                 Model              | % Gendered Words |   % Male Bias  |    F1 Score    |% Generated Responses in This Bin|
|                  :-:               |        :-:       |      :-:       |       :-:      |              :-:                |
|                Baseline            |        7.70      |      46.28     |      15.38     |             14.64               |
|                  CDA               |        7.00      |	     44.19     |      14.83     |             11.33               |
|                Pos Data            |        8.51      |	     49.71     |      15.37     |             12.28               |
|           Bias Ctrl Training       |        11.40	    |      36.41     |      15.56     |              9.62               |
|                  All               |        12.55     |      43.01     |      16.73     |             12.15               |
|            CDA + Bias Ctrl         |        11.15     |      40.89     |      15.48     |             10.32               |
|  CDA + Bias Ctrl + Our Gen. Data   |        11.54     |      44.64     |      16.61     |             12.21               |


## License
This repository is MIT licensed. See the **[LICENSE](https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/LICENSE)** file for details.
