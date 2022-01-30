# Machine Learning with PyTourch and fastai (V02)
A collection of notes, mostly from Howard and Gugger's book and teachings  [^howardandgugger-20], aimed at giving the layperson context about and an understanding of the machine learning used in this study. It is broken up into two primary subsections: ([1](#section_1)) a theoretical introduction and discussion about machine learning, and ([2](#section_2)) a deeper explanation of the methods used in this study.

## Table of Contents
* [1 Theory of Machine Learning](#section_1)
  * [1.1 Introduction to Machine Learning](#section_1_1)
  * [1.2 Advantages and Disadvantages](#section_1_2)
  * [1.3 Model Quality](#section_1_3)
  * [1.4 Model Types](#section_1_4)
  * [1.5 Model Learning](#section_1_5)
  * [1.6 Overfitting](#section_1_6)
* [2 Machine Learning Method](#section_2)
  * [2.1 Coding Environment](#section_2_1)
  * [2.2 Relevant Python Libraries](#section_2_2)
  * [2.3 Building and Deploying a Prototype with fastai and PyTorch (*Bear Classifier Computer Vision Example*)](#section_2_3)
  * [2.4 Stochastic Gradient Descent (*MNIST Handwriting Recognition Example*)](#section_2_4)
  * [2.5 Expanding SGD to Non-linearity and Building a Neural Network](#section_2_5)
  * [2.6 Other Computer Vision Problems (*PASCAL Multi-lable BIWI Regression Computer Vision Examples*)](#section_2_6)
  * [2.7 Tabular Machine Learning with Collaborative Filtering (*MovieLens Collaborative Filtering Example*)](#section_2_7)
  * [2.8 Tabular Statistical Models: Random Trees to Collaborative Filtering (*Kaggle Bulldozer Tabular Data Example*)](#section_2_8)
  * [2.9 Natural Language Processing (*IMDb Sentiment Analysis Example*)](#section_2_9)
  * [2.10 Reccurent Nueral Networks (*Human Numbers RNN Example*)](#section_2_10) 
* [3 Data Ethics](#section_3)
* [References](#references)

Version/Revision History:

Version/Revision | Date Published | Details
-----|-----|----- 
V00, Rev.01 | 2021-11-25 | Initial Draft
V01, Rev.00 | 2021-12-26 | Midterm Submission
V01, Rev.01 | 2022-01-30 | *fastai* Course Notes Added (Excluding Data Ethics) [§2.1](#section_2_1)–[.9](#section_2_9)


## 1. Theory of Machine Learning <a class="anchor" id="section_1"></a>

In this first section of the current article machine learning is introduced and its applicability and basic functioning are discussed.

### 1.1 Introduction to Machine Learning <a class="anchor" id="section_1_1"></a>

Machine learning is a widely-applicable method of problem-solving. [^howardandgugger-20] Unlike traditional engineering problem-solving methods, which require a robust understanding of the phenomenon being studied and application of definitive laws in often complex mathematical forms, machine learning allows a "black box" to be defined and trained empirically to achieve the desired result. The trained black box can then, in some cases equally or more successfully, be applied to situations where an outcome is unknown. Machine learning algorithms often take the form of neural networks, which are mathematical models which use structure (an “architecture”) and statistical weights (“parameters”) to perform complex decision making. [^howardandgugger-20] 

In the early days, before the advent of modern-day computers, the structure of neural networks had been defined, but it was difficult to apply these architectures without the ability to construct neural networks with many levels and process data in large quantities. The mathematics involved in the training are fundamentally simple since the operations involved are almost purely arithmetic; however, the iterative nature of the process makes numerical computing power necessary. [^howardandgugger-20] 

Now, machine learning has evolved to a point where it has been mathematically proven that deep learning algorithms, those with multi-level neural networks, can solve problems of infinite complexity. [^howardandgugger-20] And, the *Universal Approximation Theorem* mathematically proves that a neural network can solve any problem to any level of accuracy. [^howardandgugger-20] 

One fundamental challenge persists even with the catalyst of modern computing. That is applying data with known outcomes to a model's architecture in such a way that the results it produces for data with unknown outcomes are useful. The challenge in training neural networks is the process of finding “good” weights to do this. [^howardandgugger-20] 

### 1.2 Advantages and Disadvantages <a class="anchor" id="section_1_2"></a>

As alluded to in the previous section ([B.1.1](#section_b_1_1)), machine learning is valuable because of its ability to distill analytical problems into empirical ones. Problems that have historically required a strong conceptual understanding and mathematical characterization of the phenomena being studied, can now be solved accurately by (machine) learning the outcomes in similar situations. 

Machine learning algorithms have the ability to improve their results by adjusting weights used to generate predictions and are therefore said to have the ability to learn. [^howardandgugger-20] This is where the inherent advantage of machine learning can be observed. Instead of programming individual steps, machine learning algorithms can be trained, through experience, to achieve the same or similar results given only a set of inputs. [^howardandgugger-20]

However, Howard and Gugger [^howardandgugger-20] outline the following inherent limitations of machine learning:
* Models require labelled data (where the outcome is known) to train them.
* Models are only useful in identifying patterns that they have “seen” in their training data. 
* Models are subject to propagating any biases in their training data sets. 

Essentially, being statistical in nature, machine learning models depend entirely on the data used to train them which constitutes a significant portion of their ability to achieve desirable results (in addition to their architecture). Accordingly, any biases in the training data propagate to the model, and to its results. 

### 1.3 Model Quality <a class="anchor" id="section_1_3"></a>

A model's quality is often quantified using a metric intended for human interpretation. [^howardandgugger-20] These metrics are evaluated by comparing a model’s predictions to definitive, known results. A commonly used metric is error rate which quantifies the number of model's incorrect predictions as a percentage of all of its predictions. [^howardandgugger-20]

Labelled data refers to the dataset for which the outcomes are already known. The labels are effectively the data that the final model will be predicting for data where the outcomes are unknown. So, in order for metrics such as error rate to be evaluated, a subsection of the labelled dataset must be set aside for validation. With this subset, the model's predictions can be compared directly to the correct result or label and interpreted. For this reason, it is good practice to set aside three subsets of labelled data when developing a model for training, validation, and a final round of testing.  [^howardandgugger-20]

Interpretation of metrics evaluated with the validation set can be used to determine whether specific events have occurred in training too. One common such event is overfitting, which occurs as a result of over-training. A validation set is used to verify whether a model has been over- or under-trained. [^howardandgugger-20] The result will be an overfit or underfit which would be difficult to recognize without good metrics. Accordingly, assigning a validation set which yields accurate metrics can be nuanced. [^howardandgugger-20]  In the case of overfitting, a minimization of error in the training set may be observed, but an increase in error in the validation set which the model has not “seen” before also occurs. [^howardandgugger-20]

### 1.4 Model Types <a class="anchor" id="section_1_4"></a>

Due to the nature of machine learning, it is better at solving some problems than others. Several areas of deep learning are, computer vision, text, tabular data, recommendation systems (rec-sys), multinodal, and others. [^howardandgugger-20]

### 1.5 Model Learning <a class="anchor" id="section_1_5"></a>

A critical part of creating an effective machine learning model is training. If done correctly, the model's ability to predict outcomes will extrapolate well to data it hasn’t seen before; if done poorly, the model's predictions for unseen data will be inaccurate and unreliable.

When only a limited training set is available for a specific application, other options are available. Transfer learning, in which the model is trained on data not directly related to the data that the model will finally be applied to, is often effective. [^howardandgugger-20] To improve results, this method is followed by a fine-tuning phase where the model is trained, for several epochs, on data related directly to the final task it will be performing. [^howardandgugger-20] 

Zeiler and Fergus [^zeilerandfergus-13] have demonstrated the effectiveness of transfer learning by showing how the model they worked with learned to recognize small visual elements that occur in their training dataset and outside of it. [^howardandgugger-20] 

### 1.6 Overfitting <a class="anchor" id="section_1_6"></a>

Although it is possible to both overtrain or undertrain a model, according to Howard and Gugger [^howardandgugger-20], “the single most important and challenging issue when training for all… algorithms” is overfitting. It occurs when a model has been trained so much that its predictive capacity starts to deteriorate. [^howardandgugger-20]

Achieving correct model fit is similar to choosing a best-fit curve where increasing the degree of a polynomial may get the curve to fit the data more closely; however, at a certain point, either end of the curve might begin going off in directions not representative of the data. And, accordingly, the curve's usefulness in extrapolating the data it represents will deteriorate. There is often a fine line between a curve effectively modelling the data used to create it and accurately approximating data that wasn't used to create it. Finding this point is the challenge that also exists in avoiding underfitting or overfitting a machine learning model. 

Howard and Gugger [^howardandgugger-20] make a compelling argument to avoid overfitting models, saying that in the end, the model needs to work well on data that wasn't used to train it. 

## 2. Machine Learning Method <a class="anchor" id="section_2"></a>

There are different environments in which to practice Machine Learning. Howard and Gugger recommend a notebook that allows users to execute python code cells on a remote GPU for speed and efficiency. Regardless of the environment chosen, practical skills are equally as important as a theoretical understanding of the subject matter. In this section, some of the methods used and recommended by Howard and Gugger are discussed. 

The software and hardware proposed by Howard and Grugger, in *Deep Learning for Coders with fastai & PyTorch*, is fastai, Pythorch, and python running on a Linux computer with an NVIDIA GPU. [^howardandgugger-20] Python is a popular successor of the C and C++ programming languages. PyTorch is a “flexible and developer-friendly” tool within the language designed for machine-learning applications. [^howardandgugger-20] And, fastai is a library and API that includes many recent and useful additions to machine learning. [^howardandgugger-20] NVIDIA GPUs most commonly support deep-learning libraries. [^howardandgugger-20] And, running deep-learning applications built with this software stack on Linux machines by-passes many difficulties that may otherwise arise. [^howardandgugger-20]

fastai has four main predefined applications: [^howardandgugger-20] 
1. Vision
2. Text
3. Tabular
4. Collaborative Filtering. 

### 2.1 Coding Environemnt <a class="anchor" id="#section_2_1"></a>

Google Colab is used for model development because it is a free service that allows Jupyter Notebooks to be run on remote GPU servers. The server that executes the python scripts in code cells of a notebook is called the “kernel”, the status of which can be seen in the upper right-hand corner of a notebook. Code in each cell runs as a script, but one of the advantages of Jupyter Notebooks running on Google Colab is that data created or used in one cell can be used in another. 

In general, there are two fundamental modes in Google Colab: (1) edit mode, and (2) command mode. In edit mode, text or code can be entered into cells in the usual way. A green border around a cell indicates that the notebook is in edit mode. In command mode, keys have a specific purpose. (A few of these purposes will be discussed in the Tips subsection bellow.) A red border around a cell indicates that the notebook is in command mode. 

There are several important configuration steps required to use all of the necessary functionality and a few tips about running and editing code and text cells.

#### Configuring Google Colab

* Specify that code cells should run on a remote GPU by selecting “Runtime > Change Runtime Type” on the top menu bar and changing “Hardware Accelerator” to “GPU”.
* For any notebook using *fastai* libraries, `fastbook` must be updated and installed. This is done with two lines of code:
```python
  !pip install -Uqq fastbook
  import fastbook 
  fastbook.setup_book()
  from fastbook import *
  from fastai.vision.widgets import *
```
* Google colab allows notebooks and adjacent files to be saved in your google drive in a path that is available with the `gdrive` variable which points to `Path('content/gdrive/MyDrive')`. All adjacent files should be saved to this directory as well. 


#### Tips in Google Colab

The following is a list of a few useful commands in Google Colab:
* `esc`: puts the notebook into command mode
* `enter`: puts the notebook into edit mode
* `ctrl` + `/`: comment/uncomment (in command mode)
* `shift` + `enter`: runs the script in the current cell (in command mode)
* `h`: provides a list of keyboard shortcuts (in command mode)
* `s`: saves the notebook (in command mode)
* `up arrow` or `down arrow`: toggles between cells (in command mode)
* `b`: creates a new cell (in command mode)

### 2.2 Relevant Python Libraries <a class="anchor" id="#section_2_2"></a>

#### Tensors

Where Python, and the *NumPy* library, in particular, uses `arrays`; *PyTorch* uses `tensors`. The type is very similar but adds the ability for it to be processed on a server's GPU and not just its CPU. This is particularly useful in machine learning because of the quantity and types of computations done during training. 

Tensors are defined and indexed the same way arrays are:
```python
  data = [[1, 2, 3], [4, 5, 6]] 
  arr = array(data)
  tns = tensor(data)
```

The first item in both the array and the tensor above are indexed at `0`, so `arr[0]` and `tns[0]` respectively. To index through every column of the first row in a 2D array or tensor, `arr[1, :]` or `tns[1, :]` is the right syntax; likewise, to index through every row in the first column of a 2D array or tensor, `arr[:, 1]` or `tns[:, 1]` is the correct syntax. `arr[1, 1:3]` or `tns[1, 1:3]` is the index syntax for the second column up to but not including the fourth column in the first row, for a 2D array or tensor. The `-1` element is the last element in an array or tensor. 

Many operations can be applied to arrays and tensors, but care should be taken to distinguish between vector and term-wise operations. A few examples are as follows:
* `tns + 1` adds one to each term.
* `tns * 1.5` multiplies each term by 1.5 and changes their type to `float`.
* `tns.type()` returns the type of the entries.

#### Gradients

Howard and Gugger, inform their students that in *PyTorch* when the gradient of a variable is required, the `.reguires_grad_()` method must be called each time that variable is altered or an operation is performed on it. This method changes the type of the variable to include information about how to take its derivative. For example, consider the following:
```python
  def f(x): return x**2
    xt = tensor(3.).requires_grad_()
	 
  yt = f(xt)
  yt
```

Here, the type of `yt` is `tensor(9., grad_fn = <powBackward0>)`. And, to take the derivative with respect to `xt`, or the “gradient” the following lines of code can be used:
```python
  yt.backward()
  xt.grad()
```

This yields `tensor(6.0)` which is the derivative of `yt` with respect to `xt`, evaluated at `xt` $=3$.

#### Broadcasting 

Broadcasting allows operations to be performed on tensors with different shapes. This is done by “expand[ing] the tensor with smaller rank to have the same size as the one with larger rank”. This is powerful; however, in practice, it is recommended to check the shapes of tensors throughout the development process to confirm that variables are what is expected. 

### 2.3 Building and Deploying a Prototype with fastai and PyTorch (Bear Classifier Computer Vision Example)  <a class="anchor" id="#section_2_3"></a>

#### Data Optainment for Computer Vision

For vision models, a good way to obtain data is through Bing Image Search where batches of image search results can be downloaded at one time. To do this, a bing image search API key must be obtained from (Microsoft Azure)[https://www.microsoft.com/en-us/bing/apis/bing-web-search-api]

The API key, `###` can then be used in conjunction with the `search_images_bing` function to batch obtain URLs to image search results for a specific search term, `search term`, from the internet:
```python
  key = os.environ.get('AZURE_SEARCH_KEY', '###')
  results = search_images_bing(key, 'search term')
  ims = results.attrgot('content_url')
```

This list of 150 URLs, `ims`, can then be batch downloaded to a specific location using `download_images`. An effective workflow is to obtain the URLs and download for multiple labels all at once. This can be done using a similar scope to this one:
```python
  bear_types = 'grizzly', 'black', 'teady'
  path = Path('bears')

  if not path.exists():
    path.mkdir()
    for o in bear_types:
      dest = (path/o)
      dest.mkdir(exist_ok = True)
      results = search_images_bing(key, f'{o} bear')
      download _images(dest, urls = results.attrgota('contentUrl'))
```

Lastly, corrupt URLs and files can be deleted using the following lines of code:
```python 
  fns = get_image_files(path)
  failed = verify_images(fns)
  failed.map(Path.unlink)
```

#### Data Loading

`DataLoaders` allows a labelled dataset's type and structure to be defined within the *fastai* environment. Effectively it is a class that stores `DataLoader` objects and makes them available as `train` and `valid` for training and validation respectively.

The Data Block API is an effective way to customize all steps of creation within the data loading process from how the data is labelled and stored to how it is presented to the kernel. The API can be better understood using an example from Howard and Gugger's teaching material:
```python
  bears = DataBlock(
    blocks = (ImageBlock, CategoryBlock), 
    get_items = get_image_files, 
    splitter = RandomSplitter(valid_pct=0.2, seed=42),
    get_y = parent_label,
    item_tfms = Resize(128))
```
The `blocks=(ImageBlock, CategoryBlock)` line allows the types of the independent and dependent variables to be defined, where the independent variable is the data that will be analyzed by the completed model and the dependent variables are the labels and predictions that will be generated. 

The `get_items=get_image_files` line allows a function with which the data is taken from its source to be defined. In this case, it is a predefined function that has already been given the file path where the images can be found. 

The `splitter=RandomSplitter(valid_pct=0.2, seed=42)` line allows the quantity and location of the validation data points within the labelled dataset to be defined. In the example above, the 20% of the data in the labelled set (`valid_pct=0.2`) is set aside for validation randomly using the `RandomSplitter` and a constant randomization seed (`seed=42`) so that the validation set is repeatable between models. 

The `get_y=parent_label` line allows a function with which the label will be taken and assigned for each file to be defined. In this example, a function (`parent_label`) is used that takes the parent in the directory path (the folder the file is stored in) as the label and assigns it. This is a common method. 

Lastly, the `item_tfms=Resize(128)` line allows a format to be applied to all of the data. In this example, all images are given the same size by cropping them. 

#### Data Augmentation

In computer vision models all images must have the same format. Accordingly, `DataLoaders` has the capability of giving all images the same dimensions using several methods:
* `Resize(<x>)` gives all images `x`-pixel height and width by copping to the largest square that fits in the center of the original image. 
* `Resize(<x>, ResizeMethod.Squish)' gives all images `x`-pixel height and width by retaining the short edge of the original image and squishing the long edge to the same dimension. 
* `Resize(<x>, ResizeMethod.Pad)` gives all images `x`-pixel height and width by retaining the long edge of the original image and padding either side to give the short edge the same dimension. 
* 
Each method has its own benefits and consequences. 

The most common technique used is random data augmentation each time:
```python
  RandomResizedCrop(x, min_scale=y)  
```
With this technique, a different `x`-pixel high and wide square at least `y*100`-percent of the original image. Because the augmentation of the image changes each time the model uses it, overfitting is much less likely to occur. Hence why this technique is so popular.

#### Data Cleaning

Often it is wise to clean the labelled data after some training has been done. This is because the model has already done a relatively good job predicting the data given an imperfect labelled data set. The probabilities of an image being in a certain category can be used to pick out the images that the model was least sure about and check that they are labelled correctly.

For computer vision, *fastai* has a small GUI to clean data: 
```python
  cleaner = ImageClassifierCleaner
  cleaner
```
This GUI renders images in order from least-certain to most-certain for a given label and allows the user to delete or relabel those data points. By running the following code, these changes are saved to the data-set directly:
```python
  for idx in cleaner.delete(): cleaner.fns[idx].unlink()
  for idx, cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]),path/cat) 
```

Another common practice is to apply batch augmentations so that multiple images can be augmented at once on the GPU using `batch_tfms = aug_transforms(mult = 2)`. An example is 
```python
  Bears = bears.new(item_tfms=Resize(128), batch_tfms = aug_transforms(mult = 2))
  dls = bears.dataloaders(path)
  dls.train.show_batch(max_n=8, nrows=2, unique=True)
```
In this example, the `.new()` method creates a new instance of bears with the parameters passed in. The second and third lines are used to display the augmented data. 

#### Exporting a Model

A model can be used for inference—to generate predictions—once it has been trained, validated, and tested. At this point, it is effectively a neural network computer program with statistical weights adjusted specifically to the unique dataset it was trained on. If the model is exported in the correct form, it can be run on a server different than the one it was built and trained on. This can be beneficial for deployment.

The *fastai* and *PyTorch* methods that Howard and Gugger recommend using export both the architecture and the parameters of the model. Additionally, this method saves details about how DataLoaders was defined. So, effectively, results are made repeatable on different servers using this method. Calling `export` generates a file called *export. pkl* that contains all of this information:
```python
  learn.export()
```

#### Getting Predictions/Using the Exported Model

To use for inference (ie. to generate a prediction), an inference learner must be created using `load_learner`:
```python
  learn_inf = load_learner(path/'export.pkl')
```

An inference can then be called on a single image, `image.jpg`, using the `.predict()` method:
```python
  learn_inf.predict('path/image.jpg')
```

#### Deployment

There are two tools for Google Colab which allow graphical elements, linked to the scripts in code cells, to be added and code cells to be hidden:
IPython widgets (ipywidgets) – provide the ability to add graphical elements.
Voilà – provides the ability to hide specific code cells. 

{% include alert.html text="Add note about what lines need to be called to include these libraries" %}

Here is a list of several elements that can be added with the combination of these tools:
One useful widget that can be added with IPython is an **image upload button**:
```python
  btn_upload = widgets.FileUpload()
```
The following code is needed to store the uploaded files into an array. 
```python
  img = PILImage.creat(btn_upload.data[-1])
  img
```

**Output** widgets allow images that have been uploaded and processed to be displayed:
```python
  out_pl = widgets.Output()
  out_pl.clear_output()
  with out_pl: display(img.to_thumb(128,128))
  out_pl
```
**Lables** can be used to display text widgets with specific information:
```python
  lbl_pred = widgets.Label()
  lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
  lbl_pred
```
Other buttons, such as a **classification button** to generate a prediction and classify an image, can also be made:
```python
  btn_run = widgets.Button(description = 'Classify')
  btn_run
```
However, buttons with a specific, custom purpose need event handlers—code that runs when they are pressed. 
```python
  def on_click_classify(change):
  img = PILImage.create(btn_upload.data[-1])
  out_pl.clear_output()
  with out_pl: display(img.to_thumb(128, 128))
    pred, pred_idx, probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediciton: {pred}; Probability: {probs[pred_idx]:.04f}

  btn_run.on_click(on_click_classify
```
Widgets can be grouped vertically, using the **vertical widget grouping** functionality:
```python
  VBox([widgets.Label('Select your bear'), btn_upload, btn_run, out_pl, lbl_pred
```

####  Publishing a Prototype Application
 
Howard and Gugger recommend a method of publishing a prototype application for free using (*Binder*)[https://mybinder.org]. The benefits of this method are that it creates a web application where the inference is done on a remote server; the drawbacks are that the inference is performed on a CPU instead of a GPU, and is not ideal for processing large amounts of data in a short period.

The steps outlined in Howard and Gugger’s book are as follows:
1. “Add your notebook to a GitHub repository (*http://github.com*).
2. Past the URL of that repo into Binder’s URL field…
3. Change the FIle drop-down to instead select URL. 
4. In the “URL to open” field, enter `/voila/render/name.ipynb` (replacing `name` with the name of your notebook).
5. Click the cupboard button at the bottom right to copy the URL and past it somewhere safe. 
6. Click Launch.”

#### Safe Project Roll-Out

Howard and Gugger stress safe project rollout and suggest a framework:

1. Manual Process – During this phase, run the model in parallel with human verification.
2. Limited Scope Deployment – During this phase, deploy the model in a low-consequence subset of the final application.
3. Gradual Expansion – During this phase, expand gradually and cautiously from limited-scope deployment to full-scope deployment. 

During the roll-out process, errors should be anticipated by looking at and interpreting the data. 

One large potential issue is feedback loops where input data biases the predictions being made, and the predictions being made bias the input data is processed. There is a risk of feedback loops occurring if the model controls what the next round of data will be like. The most effective way to avoid feedback loops is by implementing human circuit breakers. Human intervention allows the data to be interpreted in a way that it cannot be by the model. And, as a result, unexpected or undesirable results can provide the impetus to stop the model and change it. 

#### Error Analysis

Some models will have an uploader:
```python
  uploader = widgets.FileUploader()
```

More information about a function can be displayed by running the following script:
```python
  Doc(<function name>)
```

The following is an example of a learner:
```python
  learn = cnn_learner(dls, resnet34, metrics = error rate)
```

`valid_pct = <x>` indicates what percentage, `x`, of the labeled data-set is reserved for validation. 

The learner is called with the following code:
```python
  learn_inf.predict(img_name)
```
`img_name` is the name of the image.

When the learner is called, a tensor indicating the probabilities of an image being in each category is returned. The tensor has the following form:
```python
  ('object', tensor(i), tensor(x, y, x, ...)
```
* `object` indicates the prediction the model made for the current incidence of the class.
* `i` indicates the index of the value in the tensor of probabilities which corresponds to the `object`.
* `x, y, x, …` is the tensor of probabilities that the current `object` is in each of the classifications.
In this example, the model predicted that the image is a grizzly bear instead of a black bear or a teddy bear:
```python
('grizzly', tensor(1), tensor([9.0767e-06, 9.9999e-01, 1.5748e-07]))
```
By looking at this data returned by the learner, it is easy to recognize that the probability of the image being a grizzly far exceeds the probabilities of it being one of the other categorizations.

The `.vocab()` method can be called to display the mapping between objects in a class and their values or for indices in an array and their values.

## 3. Data Ethics <a class="anchor" id="section_3"></a>

Not only are machine learning models subject to the same limitations of many other statistical methods—the same ethical issues that apply to statistics apply to machine learning as well. 

Howard and Gugger [^howardandgugger-20] discuss several potential issues related to interpreting statistical and probabilistic results. One conversation revolves around inappropriate uses of p-value analysis and hypothesis testing. 

According to the American Statistical Association [^asa-16], p-value analysis can be inappropriate for the following reasons: [^howardandgugger-20] 
* P-values can indicate incompatibility between a data set and a specific statistical model. 
* P-values do not probabilistically distinguish the event that the studied hypothesis was true from the event that a random coronation was observed in the data.
* Scientific conclusions, business or policy decisions should not be made based on whether a p-value surpasses a specific threshold. 
* Proper reporting depends on inference for which full transparency is necessary. 
* P-values and statistical significance do not quantify the size or importance of an observed effect.
* Alone, a p-value does not provide a good measure of evidence regarding a model or hypothesis.

Conclusions made with p-value analysis are dependent entirely on the data that analysis was performed on, and are accordingly subject to the same biases. [^howardandgugger-20] 

Howard and Gugger [^howardandgugger-20] provide a “strategy to achieving valuable statistical results” in Appendix B of their book. However, two notable assertions can be inferred from their discussion about appropriate and inappropriate uses of p-value analysis:
1. Statistical results should be validated by comparing them to actual, practical outcomes. [^howardandgugger-20] 
2. One of the following classifications should be assigned and all possibilities should be considered along with any statistical or probabilistic conclusion: [^howardandgugger-20]

-----|-----
There is no real relationship, but act as if there is one|There is a real relationship, and act as if there is one
There is no real relationship, and act as if there isn’t one|There is no real relationship, and act as if there is one

## References <a class="anchor" id="references"></a>
[^evins-21]: Evins, 2021: *Building Energy Data for Machine Learning*
[^howardandgugger-20]: Howard and Gugger, 2021: *Deep Learning for Coders with fastai & PyTorch*
[^zeilerandfergus-13]: Zeiler and Fergus, 2013: *Visualizing and Understanding Convolutional Networks*
[^asa-16]: ASA, 2016: *American Statistical Association Releases Statement on Statistical Significance and P-Values*