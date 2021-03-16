# Mitigating-Gender-Bias-in-Generated-Text

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
Below are the results from reproducing the experiments done by Dinan et al. and for extensions to these experiments. The experiment extensions are both aimed at addressing the high time and monetary cost of positively biased data collection, which requires crowdsourcing data. The first of these extensions is fine-tuning the pretrained Reddit model on the data generated from counterfactual data augmentation and using bias controlled training to determine the impact of excluding positively biased data collection. The second extension still fine-tunes the model using counterfactual data augmentation and bias controlled training, but also includes neutral data we generate (the process for generating this data is described in the "Neutral Generated Data" section above). The results below give the percent gendered words (number of gendered words out of the total number of words in the generated responses), percent male bias (number of male gendered words out of the gendered words), and F1 score for each model for four bins: F0M0, F+M0, F0M+, and F+M+. The test data is split into bins based on the presence of gendered words in the label (the next response in the conversation). F0M0 means there are no gendered words in the label. F+M0 means there is at least one female gendered word but no male gendered words in the label. F0M+ means there are no female gendered words but at least one male gendered word in the label. F+M+ means there is at least one female and one male gendered word in the label.

The image below shows how each bias mitigation technique is used to mitigate gender bias in the generated text. The plots separate the data into buckets used for bias controlled training to show how these techniques mitigate bias in the generated text. This plot also shows how well bias controlled training gives control to the model when generating text by telling the model what type of data it must generate via passing the bucket as part of the features in the episode.

<p align="center">
  <img src="https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/images/ReproducibilityChartResults.PNG">
  <b>Results for Reproducing the Experiments in Original Paper by <a href="https://arxiv.org/abs/1911.03842">Dinan et al.</a></b><br>
  <br><br>  
</p>

### Results for F+M0 Bin

### Results for F0M+ Bin

### Results for F+M+ Bin

## License
This repository is MIT licensed. See the **[LICENSE](https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/LICENSE)** file for details.
