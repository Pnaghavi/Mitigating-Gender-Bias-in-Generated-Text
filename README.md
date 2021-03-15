# Mitigating-Gender-Bias-in-Generated-Text

This repository is setup to provide access to the code and data which was used in writing the paper [Reproducing and extending “Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation”](http://parl.ai). The original paper [Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation](https://arxiv.org/abs/1911.03842) utilizes [ParlAI](https://parl.ai/) for training, validating, and text generation. The model used in these papers is a 87M parameter transformer with 16 attention heads, 8 encoder layers, 8 decoder layers, and embedding size of 512. This model is pretrained on the [large Reddit](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) dataset  with approximately 1.7 billion comments. In all experiments conducted in this paper this pretrained model was fine tuned on the [Light](https://parl.ai/projects/light/#:~:text=The%20original%20LIGHT%20dataset%20features,interactions%20(talking%20and%20acting).) dataset with some form of data augmentation or training to mitigate the gender bias noticeable in the conversations in the game. In large corpora there is potential for text containing gender and racial bias, which the models can learn from and generate racially and gender biased text. The goal of these papers is to mitigate gender bias present in corpora and prevent model from generating gender biased text. In order to accomplish this task, the original paper came up with three bias mitigation techniques.
-	**Counterfactual Data Augmentation**
This augmentation method uses a set of gendered words and replaces them with their opposite. For example, it swaps king for queen or priest for priestess.
-	**Positive-Bias Data Collection**
This method utilizes neutral crowd sourced data to shift the gender bias to a more neutral level. Although the amount of data infusion is about 10% of the original dataset it has huge effect on shifting the bias present in the dataset.
-	**Bias Controlled Training**
This method puts each label into a bucket based on the type of bias present in the text. For example, if the text includes male gendered words but no female it is assigned to f0 m1 bucket, but if both female and male gendered words are present it assigns it to f1 m1 bucket. The bucket labels are added to the end or training features for the model to learn from when generating a response.
-	**Positive-Bias Generated Data**
This method is an extension to the original paper. For this method we originally train a model using counterfactual data augmentation and bias control training and use it generate responses to the entire training data. Then we go through the generated text and 90% of the time pick the generated responses that are neutral such as those belonging to f0 m0 bucket or f1 m1 with relatively equal number of male and female gendered words present in them. We use the neutral generated responses to reconstruct the conversations and use the neutral conversation to train a new model.
All these bias mitigation techniques are very useful in mitigating gender bias in generated text.


## Interactive Tutorial

For those who want to start with ParlAI now, you can try our [Colab Tutorial](https://colab.research.google.com/drive/1bRMvN0lGXaTF5fuTidgvlAl-Lb41F7AD#scrollTo=KtVz5dCUmFkN).

## Installing ParlAI

ParlAI currently requires Python3.7+ and [Pytorch](https://pytorch.org) 1.6 or higher.
Dependencies of the core modules are listed in [`requirements.txt`](https://github.com/facebookresearch/ParlAI/blob/master/requirements.txt). Some
models included (in [`parlai/agents`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents)) have additional requirements.
We *strongly* recommend you install ParlAI in a [venv](https://docs.python.org/3/library/venv.html) or [conda](https://www.anaconda.com/) environment.

**Standard Installation**

If you want to use ParlAI without modifications, you can install it with:

```bash
pip install parlai
```

**Development Installation**

Many users will want to modify some parts of ParlAI. To set up a development
environment, run the following commands to clone the repository and install
ParlAI:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
```

All needed data will be downloaded to `~/ParlAI/data`. If you need to clear out
the space used by these files, you can safely delete these directories and any
files needed will be downloaded again.

## Documentation

 - [Quick Start](https://parl.ai/docs/tutorial_quick.html)
 - [Basics: world, agents, teachers, action and observations](https://parl.ai/docs/tutorial_basic.html)
 - [Creating a new dataset/task](http://parl.ai/docs/tutorial_task.html)
 - [List of available tasks/datasets](https://parl.ai/docs/tasks.html)
 - [Creating a seq2seq agent](https://parl.ai/docs/tutorial_torch_generator_agent.html)
 - [List of available agents](https://parl.ai/docs/agents_list.html)
 - [Model zoo (list pretrained models)](https://parl.ai/docs/zoo.html)
 - [Running crowdsourcing tasks](http://parl.ai/docs/tutorial_crowdsourcing.html)
 - [Plug into Facebook Messenger](https://parl.ai/docs/tutorial_chat_service.html)


## Examples

A large set of scripts can be found in [`parlai/scripts`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/scripts). Here are a few of them.
Note: If any of these examples fail, check the [installation section](#installing-parlai) to see if you have missed something.

Display 10 random examples from the SQuAD task
```bash
parlai display_data -t squad
```

Evaluate an IR baseline model on the validation set of the Personachat task:
```bash
parlai eval_model -m ir_baseline -t personachat -dt valid
```

Train a single layer transformer on PersonaChat (requires pytorch and torchtext).
Detail: embedding size 300, 4 attention heads,  2 epochs using batchsize 64, word vectors are initialized with fasttext and the other elements of the batch are used as negative during training.
```bash
parlai train_model -t personachat -m transformer/ranker -mf /tmp/model_tr6 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 2 -veps 0.25 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch
```

## Code Organization

The code is set up into several main directories:

- [**core**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/core): contains the primary code for the framework
- [**agents**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents): contains agents which can interact with the different tasks (e.g. machine learning models)
- [**scripts**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/scripts): contains a number of useful scripts, like training, evaluating, interactive chatting, ...
- [**tasks**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks): contains code for the different tasks available from within ParlAI
- [**mturk**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk): contains code for setting up Mechanical Turk, as well as sample MTurk tasks
- [**messenger**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/chat_service/services/messenger): contains code for interfacing with Facebook Messenger
- [**utils**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/utils): contains a wide number of frequently used utility methods
- [**crowdsourcing**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing): contains code for running crowdsourcing tasks, such as on Amazon Mechanical Turk
- [**chat_service**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/chat_service/services/messenger): contains code for interfacing with services such as Facebook Messenger
- [**zoo**](https://github.com/facebookresearch/ParlAI/tree/master/parlai/zoo): contains code to directly download and use pretrained models from our model zoo

## Support
If you have any questions, bug reports or feature requests, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/ParlAI/issues).
You may also be interested in checking out our [FAQ](https://parl.ai/docs/faq.html) and
our [Tips n Tricks](https://parl.ai/docs/tutorial_tipsntricks.html).

Please remember to follow our [Code of Conduct](https://github.com/facebookresearch/ParlAI/blob/master/CODE_OF_CONDUCT.md).

## Contributing
We welcome PRs from the community!

You can find information about contributing to ParlAI in our
[Contributing](https://github.com/facebookresearch/ParlAI/blob/master/CONTRIBUTING.md)
document.


## The Team
ParlAI is currently maintained by Moya Chen, Emily Dinan, Dexter Ju, Mojtaba
Komeili, Spencer Poff, Pratik Ringshia, Stephen Roller, Kurt Shuster,
Eric Michael Smith, Megan Ung, Jack Urbanek, Jason Weston, Mary Williamson,
and Jing Xu. Stephen Roller is the current Tech Lead.

Former major contributors and maintainers include Alexander H. Miller, Margaret
Li, Will Feng, Adam Fisch, Jiasen Lu, Antoine Bordes, Devi Parikh, Dhruv Batra,
Filipe de Avila Belbute Peres, Chao Pan, and Vedant Puri.

## Citation

Please cite the [arXiv paper](https://arxiv.org/abs/1705.06476) if you use ParlAI in your work:

```
@article{miller2017parlai,
  title={ParlAI: A Dialog Research Software Platform},
  author={{Miller}, A.~H. and {Feng}, W. and {Fisch}, A. and {Lu}, J. and {Batra}, D. and {Bordes}, A. and {Parikh}, D. and {Weston}, J.},
  journal={arXiv preprint arXiv:{1705.06476}},
  year={2017}
}
```

## License
ParlAI is MIT licensed. See the **[LICENSE](https://github.com/facebookresearch/ParlAI/blob/master/LICENSE)** file for details.