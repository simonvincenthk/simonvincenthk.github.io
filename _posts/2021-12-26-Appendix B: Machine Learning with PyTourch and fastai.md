# Appendix B: Machine Learning with PyTourch and fastai
An appendix to *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture*

This appendix is a collection of notes, mostly from Howard and Gugger’s book and teachings  [^howardandgugger-20], aimed at giving the layperson context about and an understanding of the machine learning used in this study. It is broken up into two primary subsections: (B.1) a theoretical introduction and discussion about machine learning, and (B.2) a deeper explanation of the methods used in this study.

## Table of Contents
* [B.1 Theory of Machine Learning](#section_b_1)
  * [B.1.1 Introduction to Machine Learning](#section_b_1_1)
  * [B.1.2 Advantages and Disadvantages](#section_b_1_2)
  * [B.1.3 Model Quality](#section_b_1_3)
  * [B.1.4 Model Types](#section_b_1_4)
  * [B.1.5 Model Learning](#section_b_1_5)
  * [B.1.6 Overfitting](#section_b_1_6)
* [B.2 Machine Learning Method](#section_b_2)
* [B.3 Data Ethics](#section_b_3)
* [References Revisited](#references-revisited)


Version/Revision History:

Version/Revision | Date Published | Details
-----|-----|----- 
V00, Rev.01 | 2021-11-25 | Initial Draft
V01, Rev.00 | 2021-12-26 | Midterm Submission

## B.1 Theory of Machine Learning <a class="anchor" id="section_b_1"></a>

In this first section of the current appendix (B) machine learning is introduced and its applicability and basic functioning are discussed.

### B.1.1 Introduction to Machine Learning <a class="anchor" id="section_b_1_1"></a>

Machine learning is a widely-applicable method of problem-solving. [^howardandgugger-20] Unlike traditional engineering problem-solving methods, which require a robust understanding of the phenomenon being studied and application of definitive laws in often complex mathematical forms, machine learning allows a "black box" to be defined and trained empirically to achieve the desired result. The trained black box can then, in some cases equally or more successfully, be applied to situations where an outcome is unknown. Machine learning algorithms often take the form of neural networks, which are mathematical models which use structure (an “architecture”) and statistical weights (“parameters”) to perform complex decision making. [^howardandgugger-20] 

In the early days, before the advent of modern-day computers, the structure of neural networks had been defined, but it was difficult to apply these architectures without the ability to construct neural networks with many levels and process data in large quantities. The mathematics involved in the training are fundamentally simple since the operations involved are almost purely arithmetic; however, the iterative nature of the process makes numerical computing power necessary. [^howardandgugger-20] 

Now, machine learning has evolved to a point where it has been mathematically proven that deep learning algorithms, those with multi-level neural networks, can solve problems of infinite complexity. [^howardandgugger-20] And, the Universal Approximation Theorem mathematically proves that a neural network can solve any problem to any level of accuracy. [^howardandgugger-20] 

One fundamental challenge persists even with the catalyst of modern computing. That is applying data with known outcomes to a model's architecture in such a way that the results it produces for data with unknown outcomes are useful. The challenge in training neural networks is the process of finding “good” weights to do this. [^howardandgugger-20] 

### B.1.2 Advantages and Disadvantages <a class="anchor" id="section_b_1_2"></a>

As alluded to in the previous section (B.1.1), machine learning is valuable because of its ability to distill analytical problems into empirical ones. Problems that have historically required a strong conceptual understanding and mathematical characterization of the phenomena being studied, can now be solved accurately by (machine) learning the outcomes in similar situations. 

Machine learning algorithms have the ability to improve their results by adjusting weights used to generate predictions and are therefore said to have the ability to learn. [^howardandgugger-20] This is where the inherent advantage of machine learning can be observed. Instead of programming individual steps, machine learning algorithms can be trained, through experience, to achieve the same or similar results given only a set of inputs. [^howardandgugger-20]

However, Howard and Gugger [^howardandgugger-20] outline the following inherent limitations of machine learning:
* Models require labelled data (where the outcome is known) to train them.
* Models are only useful in identifying patterns that they have “seen” in their training data. 
* Models are subject to propagating any biases in their training data sets. 

Essentially, being statistical in nature, machine learning models depend entirely on the data used to train them which constitutes a significant portion of their ability to achieve desirable results (in addition to their architecture). Accordingly, any biases in the training data propagate to the model, and to its results. 

### B.1.3 Model Quality <a class="anchor" id="section_b_1_3"></a>

A model's quality is often quantified using a metric intended for human interpretation. [^howardandgugger-20] These metrics are evaluated by comparing a model’s predictions to definitive, known results. A commonly used metric is error rate which quantifies the number of model's incorrect predictions as a percentage of all of its predictions. [^howardandgugger-20]

Labelled data refers to the dataset for which the outcomes are already known. The labels are effectively the data that the final model will be predicting for data where the outcomes are unknown. So, in order for metrics such as error rate to be evaluated, a subsection of the labelled dataset must be set aside for validation. With this subset, the model's predictions can be compared directly to the correct result or label and interpreted. For this reason, it is good practice to set aside three subsets of labelled data when developing a model for training, validation, and a final round of testing.  [^howardandgugger-20]

Interpretation of metrics evaluated with the validation set can be used to determine whether specific events have occurred in training too. One common such event is overfitting, which occurs as a result of over-training. A validation set is used to verify whether a model has been over- or under-trained. [^howardandgugger-20] The result will be an overfit or underfit which would be difficult to recognize without good metrics. Accordingly, assigning a validation set which yields accurate metrics can be nuanced. [^howardandgugger-20]  In the case of overfitting, a minimization of error in the training set may be observed, but an increase in error in the validation set which the model has not “seen” before also occurs. [^howardandgugger-20]

### B.1.4 Model Types <a class="anchor" id="section_b_1_4"></a>

Due to the nature of machine learning, it is better at solving some problems than others. Several areas of deep learning are, computer vision, text, tabular data, recommendation systems (rec-sys), multinodal, and others. [^howardandgugger-20]

### B.1.5 Model Learning <a class="anchor" id="section_b_1_5"></a>

A critical part of creating an effective machine learning model is training. If done correctly, the model's ability to predict outcomes will extrapolate well to data it hasn’t seen before; if done poorly, the model's predictions for unseen data will be inaccurate and unreliable.

When only a limited training set is available for a specific application, other options are available. Transfer learning, in which the model is trained on data not directly related to the data that the model will finally be applied to, is often effective. [^howardandgugger-20] To improve results, this method is followed by a fine-tuning phase where the model is trained, for several epochs, on data related directly to the final task it will be performing. [^howardandgugger-20] 

Zeiler and Fergus [^zeilerandfergus-13] have demonstrated the effectiveness of transfer learning by showing how the model they worked with learned to recognize small visual elements that occur in their training dataset and outside of it. [^howardandgugger-20] 

### B.1.6 Overfitting <a class="anchor" id="section_b_1_6"></a>

Although it is possible to both overtrain or undertrain a model, according to Howard and Gugger [^howardandgugger-20], “the single most important and challenging issue when training for all… algorithms” is overfitting. It occurs when a model has been trained so much that its predictive capacity starts to deteriorate. [^howardandgugger-20]

Achieving correct model fit is similar to choosing a best-fit curve where increasing the degree of a polynomial may get the curve to fit the data more closely; however, at a certain point, either end of the curve might begin going off in directions not representative of the data. And, accordingly, the curve's usefulness in extrapolating the data it represents will deteriorate. There is often a fine line between a curve effectively modelling the data used to create it and accurately approximating data that wasn’t used to create it. Finding this point is the challenge that also exists in avoiding underfitting or overfitting a machine learning model. 

Howard and Gugger [^howardandgugger-20] make a compelling argument to avoid overfitting models, saying that in the end, the model needs to work well on data that wasn’t used to train it. 

## B.2 Machine Learning Method <a class="anchor" id="section_b_2"></a>

The software and hardware proposed by Howard and Grugger, in Deep Learning for Coders with fastai & PyTorch, is fastai, Pythorch, and python running on a Linux computer with an NVIDIA GPU. [^howardandgugger-20] Python is a popular successor of the C and C++ programming languages. PyTorch is a “flexible and developer-friendly” tool within the language designed for machine-learning applications. [^howardandgugger-20] And, fastai is a library and API that includes many recent and useful additions to machine learning. [^howardandgugger-20] NVIDIA GPUs most commonly support deep-learning libraries. [^howardandgugger-20] And, running deep-learning applications built with this software stack on Linux machines by-passes many difficulties that may otherwise arise. [^howardandgugger-20]

fastai has four main predefined applications: [^howardandgugger-20] 
1. Vision
2. Text
3. Tabular
4. Collaborative Filtering. 

## B.3 Data Ethics <a class="anchor" id="section_b_3"></a>

Not only are machine learning models subject to the same limitations of many other statistical methods—the same ethical issues that apply to statistics apply to machine learning as well. 

Howard and Gugger [^howardandgugger-20] discuss several potential issues related to interpreting statistical and probabilistic results. One conversation revolves around inappropriate uses of p-value analysis and hypothesis testing. 

According to the American Statistical Association [^asa-16], p-value analysis can be inappropriate for the following reasons [^howardandgugger-20]: 
* P-values can indicate incompatibility between a data set and a specific statistical model. 
* P-values do not probabilistically distinguish the event that the studied hypothesis was true from the event that a random coronation was observed in the data.
* Scientific conclusions, business or policy decisions should not be made based on whether a p-value surpasses a specific threshold. 
* Proper reporting depends on inference for which full transparency is necessary. 
* P-values and statistical significance do not quantify the size or importance of an observed effect.
* Alone, a p-value does not provide a good measure of evidence regarding a model or hypothesis.

Conclusions made with p-value analysis are dependent entirely on the data that analysis was performed on, and are accordingly subject to the same biases [^howardandgugger-20]. 

Howard and Gugger [^howardandgugger-20] provide a “strategy to achieving valuable statistical results” in Appendix B of their book. However, two notable assertions can be inferred from their discussion about appropriate and inappropriate uses of p-value analysis:
Statistical results should be validated by comparing them to actual, practical outcomes [^howardandgugger-20]. 
One of the following classifications should be assigned and all possibilities should be considered along with any statistical or probabilistic conclusion [^howardandgugger-20]: 

|:-----|:-----|
|There is no real relationship, but act as if there is one.|There is a real relationship, and act as if there is one.|
|There is no real relationship, and act as if there isn’t one.|There is no real relationship, and act as if there is one.|

## References Revisited <a class="anchor" id="references-revisited"></a>
[^evins-21]: Evins, 2021: *Building Energy Data for Machine Learning*
[^howardandgugger-20]: Howard and Gugger, 2021: *Deep Learning for Coders with fastai & PyTorch*
[^zeilerandfergus-13]: Zeiler and Fergus, 2013: *Visualizing and Understanding Convolutional Networks*
[^asa-16]: ASA, 2016: *American Statistical Association Releases Statement on Statistical Significance and P-Values*

