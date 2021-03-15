# Mitigating-Gender-Bias-in-Generated-Text

This repository is setup to provide access to the code and data which was used in writing the paper [Reproducing and extending “Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation”](http://parl.ai). The original paper [Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation](https://arxiv.org/abs/1911.03842) utilizes [ParlAI](https://parl.ai/) for training, validating, and text generation. The model used in these papers is a 87M parameter transformer with 16 attention heads, 8 encoder layers, 8 decoder layers, and embedding size of 512. This model is pretrained on the [large Reddit](https://pushshift.io/) dataset  with approximately 1.7 billion comments. In all experiments conducted in this paper this pretrained model was fine tuned on the [Light](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) dataset with some form of data augmentation or training to mitigate the gender bias noticeable in the conversations in the game. In large corpora there is potential for text containing gender and racial bias, which the models can learn from and generate racially and gender biased text. The goal of these papers is to mitigate gender bias present in corpora and prevent model from generating gender biased text. In order to accomplish this task, the original paper came up with three bias mitigation techniques.
-	**Counterfactual Data Augmentation**\
This augmentation method uses a set of gendered words and replaces them with their opposite. For example, it swaps king for queen or priest for priestess.
-	**Positive-Bias Data Collection**\
This method utilizes neutral crowd sourced data to shift the gender bias to a more neutral level. Although the amount of data infusion is about 10% of the original dataset it has huge effect on shifting the bias present in the dataset.
-	**Bias Controlled Training**\
This method puts each label into a bucket based on the type of bias present in the text. For example, if the text includes male gendered words but no female it is assigned to f0 m1 bucket, but if both female and male gendered words are present it assigns it to f1 m1 bucket. The bucket labels are added to the end or training features for the model to learn from when generating a response.
-	**Positive-Bias Generated Data**\
This method is an extension to the original paper. For this method we originally train a model using counterfactual data augmentation and bias control training and use it generate responses to the entire training data. Then we go through the generated text and 90% of the time pick the generated responses that are neutral such as those belonging to f0 m0 bucket or f1 m1 with relatively equal number of male and female gendered words present in them. We use the neutral generated responses to reconstruct the conversations and use the neutral conversation to train a new model.
All these bias mitigation techniques are very useful in mitigating gender bias in generated text.


## Original Paper's Github Repository 

The original paper Dinan et al. uses the ParlAI infrastructure to run the experiments. The [genderation bias github repository](https://github.com/facebookresearch/ParlAI/tree/master/projects/genderation_bias) gives the necessary information on the paper. The general [Facebook research ParlAI bias github repository](https://github.com/facebookresearch/ParlAI) give information about pretrained model and datsets used for various research projects including [Light](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) dataset.

## Reproduction and Extension Code

The entire [source code](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) is documented and sectioned for ease of access in an ipython notebook. Each section in the notebook includes a particular experiment's code and result described in the paper. In addition, extension code and results are provided in the ipython notebook. The code is organized in such a way that to run the notebook for the first time ever, you need run the initial setup and then the experiment code starting with the general training subsection, but all subsequent times after the first time the regular setup can be used followed by the desired experiment cell. This code is best suited for running on google drive with google collaboratory with access to [PyTorch](https://pytorch.org/) compatible GPU, but with minor modifications it can be run a regular or other virtual machines. More description on how to use the code is provided on the ipython notebook provided in the src folder.

## Dependencies
The dependencies for this project are all met when running the ipython notebook on google drive with google collaboratory with access to [PyTorch](https://pytorch.org/) compatible GPU, but if this is not the machine you are using, you need the following dependencies and their subsequent dependencies to be met:
-   [PyTorch](https://pytorch.org/) with access to a compatible GPU
-   [ParlAI](https://parl.ai/)
-   [subword-nmt](https://github.com/rsennrich/subword-nmt)
-   [Numpy](https://numpy.org/)
-   [NLTK](https://www.nltk.org/)
Additionally, general [python 3.7.10](https://www.python.org/downloads/release/python-3710/) libraries need to be installed such re, copy, os, json, pickle, random, sys.

## Data Download Instructions
To download the data please install [ParlAI](https://parl.ai/) using:\
```bash
pip install parlai
```
Once ParlAI is installed on the machine using the command below from the [paper's ParlAI page](https://parl.ai/projects/genderation_bias/) all the data from the original paper can be downloaded.\
```bash
parlai display_data -t light_genderation_bias
```
Also, that data is available on the data folder of this repository for ease of access.

## Pretrained Model Download
The pretrained model is automatically downloaded when using the code provided in the ipython notebook in the src folder, but the following command will also download the pretrained model trained the Reddit dataset used in this paper.
```bash
parlai interactive -mf zoo:tutorial_transformer_generator/model 
```
## License
This repository is MIT licensed. See the **[LICENSE](https://github.com/Pnaghavi/Mitigating-Gender-Bias-in-Generated-Text/blob/main/LICENSE)** file for details.