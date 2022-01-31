# Machine Learning with PyTourch and fastai (V02)
A collection of notes, mostly from Howard and Gugger's book and teachings [^howardandgugger-20], for the author's personal used but whih may also give the layperson context about and an understanding of the machine learning. The artilcle is broken up into two primary subsections: ([1](#section_1)) a theoretical introduction and discussion about machine learning, and ([2](#section_2)) a deeper explanation of the methods used in this study.

{% include alert.html text="Please note that for the current version of this article, citations are neither consistant nor complete. However all-uncited writing is derived from Howard and Gugger's [^howardandgugger-20](*fastai* "Practical Deep Learning for Coders" course) and book." %}

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
V01, Rev.01 | 2022-01-30 | *fastai* Course Notes Added (Excluding Data Ethics) §[2.1](#section_2_1)–[.10](#section_2_10)


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

### 2.4 Stochastic Gradient Descent (MNIST Handwriting Recognition Example) <a class="anchor" id="#section_2_4"></a>

#### Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent (SGD) is a foundational numerical optimization algorithm used in machine learning. 

The previous example, where the objective was to distinguish a 3 from a 7, can be approached differently. Instead of programming a step-by-step process to compare undetermined images to two ideals, a function computing the probability that each pixel will be dark or light can be programmed and optimized through stochastic gradient descent. 

For the purposed of this exercise, that function is defined as follows:
```python
  def pr-eight(x,w) = (x*w).sum()
```

Where `x` is a vector of probabilities corresponding to each pixel, and `w` is a vector of weights corresponding to each pixel. 

The gradient descent method applied to this problem is defined by Howard and Grugger, follows:
1. Initialize weights.
2. For each image, use the weights to predict whether a number appears to be a three or a seven.
3. Based on these predictions, calculate how good the model is (ie. its loss).
4. Calculate the gradient, which measures for each weight how changing the weight would change the loss. 
5. Step (that is, change) all the weights based on that calculation.
6. Repeat.
7. Stop, when a desirable loss is reached. 

There are many methods to approach each step.

#### Data Fitting Example of Stochastic Gradient Descent

Consider the following example which uses gradient descent to fit a curve to a plot of roller coast speeds with respect to time. It is known that the best fit curve will have the form $v\left(t\right) = at^{2} + bt + c$. So, the task is to find the constants $a$, $b$, and $c$ that cause the function to best match the data. 

The function is defined with the following lines of code:
```python
  def ft(t, params):
    a, b, c = params
    return a*(t**2) + (b*t) + c
```

A loss function is defined as follows:

```python
  def mse(preds, targets):
    return((preds - targets) ** 2).mean()
```

Next, an initial set of parameters:

```python
  params = touch_randn(3).requires_grad_()
  orig_params = params.clone()
```

Now, with the parameters initialized and a function and a loss function defined, the four recursive steps of forming a prediction, calculating an associate loss, calculating an associated gradient, and stepping can be performed:

```python
  def apply_step(params, prn = TRUE):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr*params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds

  for i in range(10): apply_step(params)
```

Note that the use of the `.data` method instructs the compiler to only perform the computation on the value as opposed to the information about the variable's gradient.

#### MNIST Handwriting Recognition Problem

A classic problem in machine learning is hand-writing recognition. A simplified version is distinguishing between photos of a hand-written “3” and a hand-written “7”. 

Without machine learning, one might try to approach the problem with an algorithm. For example, taking a set of images of threes with the same pixel dimensions, and averaging each pixel's darkness across the data-set to come up with a baseline to compare unlabeled images to and doing the same for a set of images of sevens. By taking the mean absolute difference (L1 Norm) or the root mean squared error (L2 Norm) one could make an assertion about which the unknown image is more similar to.

Oftentimes a baseline algorithm of this form can be constructed and used to compare a machine-learning algorithm to. If it is more effective than the algorithm, progress has been made. 

Stochastic gradient descent can be applied to this problem by applying it to the probabilities of each pixel being dark:
```python
  w -= gradient(w) * lr
```

Where `lr` is the “learning rate”. A good choice of learning rate can impact both whether a problem can be solved and how fast it is solved. 

#### Applying Stochastic Gradient Descent to the MNIST Handwriting Recognition Example

The tensor form of each image is reshaped into a vector (a 1D tensor) using the `.view()` method:

```python
  train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
```

Note that using `-1` as the parameter for rows in the `.view()` method indicates that as many rows as are needed for the data will be assigned. It is known that each image is 28 by 28 pixels, so there will be one row. Effectively, this line of code concatenates all images of threes into one matrix and all images of sevens into one matrix where each `28*28`-long row represents one image. 

Next, labels are assigned:
```python
  train_y = tensor([1]* len(thress) + [0]*len(sevens)).unsqueeze(1)
```

A `1` is used for each of the threes and a `0` is used for each of the sevens. The `.unsqueeze()` method converts `train_y` from a vector (one row and many columns) to a matrix with one column and 12396 rows. 

Calling `train_x.shape, train_y.shape` returns the shape of both matrices:
```
  (torch.size([12396, 748]), torch.size([12396, 1]))
```

Together `train_x` and `train_y` can be combined into a dataset, `dset`, which is a list of corresponding inputs and outputs:
```python
  dset = list(zip(train_x, train_y))
  x, y = dset[0]
```

In the second line of the code above, the dataset, `dset`, is destructured. 

Now, the parameters are initialized to random values:
```python
def init_params(size, std = 1.0): return (torch.randn(size)*std).requires_grad_()
  weights = init_params(28*28, 1)
```

The function for the parameters, `y`, is linear and must have the form `y = w*x + b`, where `w` is the “weight” and `b` is a bias so that `y` will not equal zero when the pixels are equal to zero. The bias is implemented as follows:
```python
  bias = init_params(1)
```

A prediction for each image can now be generated. The operations necessary to perform this computation are done quickly by performing matrix multiplication with the `@` operator. This is done on the GPU instead of using for loops to calculate it on the CPU.

```python
  def linear1(xb): return xb@weights + bias
  preds = linear1(train_x)
```

This linear function of the form `batch@wegihts + bias` is one of the fundamental equations for any neural network. The other is the so-called “activation function”. The activation will assign the label three (`True`) to a prediction greater than zero and the label seven (`False`)  to a prediction less than zero:
```python
  corrects = (preds>0.0).float() = train_y
```

By taking the mean of the boolean `is three` values assigned for each image, the percentage of times that the model predicts a three can be seen:
```python
  corrects.float().mean().item()
```

Here, the `.item()` method displays more decimal places. 

The gradient of the predictions with respect to the weights can be taken, by calculating the difference in parameters for a small difference in weights. In other words, the slope, $m$ is defined as

$$m = \dfrac{y_{\text{final}}-y_{\text{final}}}{x_{\text{final}}-x_{\text{final}}}$$

Implemented, for the first weight and prediction that looks like this:
```python
  weights[0] *= 1.0001
  preds = linear1(train_x)
  ((preds>0.0).float() == train_y).float().mean().item()
```

No change in prediction is noticeable in any of the decimal places. Due to the threshold assigned above, where predictions greater than zero are threes and predictions less than zero are sevens, the loss function is stepped and has a zero gradient almost everywhere. So, a new loss function must be defined. To use the gradient effectively, a slightly better prediction must have a slightly better loss. 

A loss function can be defined as the difference between the target (`1` or `0`) and the probabilistic prediction of a data point being a `1` or `0`:
```python
  def mnist_loss(predictions, targets):
    return torch.where(targets == 1, 1 - predictions, predictions).mean()
```

This loss is the smallest for good performance of the system. It is generally better for a gradient descent method because it is not a stepped function with many areas of zero slope.

However, in practice, one issue remains. The predictions are not normalized to be in the range of zero to one. To do this the Sigmoid function, which is defined as follows, is used:
```python
  def sigmoid(x): return 1/(1+torch.exp(-x))
```

Combined, the final loss function has the following form: 
```python
  def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()
```

Notice that the sigmoid function is a defined method within the included libraries. 

The accuracy, which is what matters, is effectively the inverse of the loss, where it has a high value for good performance and a low value for poor performance. Accuracy is known as a metric, because it is for human interpretation, whereas the loss is used by the program and must have a “nicely behaved gradient”.

The gradient should be used to update the weights of the prediction function in batches, using the GPU for the sake of speed. This is done using matrix multiplication operations which can be performed on the GPU, instead of iterative operations performed on the CPU. A mini-batch approach referred to processing a few pieces of data at a time on the GPU. The size of this mini-batch is called batch size. There is a happy medium in choosing batch sizes where each batch and the number of batches are balanced to minimize the overall run time. 

The batch size is defined using the DataLoader function:
```python 
  coll = range(15)
  dl = DataLoader(coll, batch_size = 5, shuffle = True)
  list(dl)
```

DataLoader returns an iterator which is assigned to `dl` in the previous lines of code. Calling `list` with `dl` passed as a parameter lists all five items. In the above example `coll` is a collection of items from zero to 15, `batch_size = 5` assigns a batch size of five items, and `shuffle = True` tells DataLoader to shuffle the items. Note that random orders are good because it avoids the model learning the dataset as opposed to the data itself. 

When a tuple is processed using DataLoader, it creates tuples of batches. So `[[1, a], [2, b], [3, c]` would be processed to `[[1, 2, 3], [a, b, c]]`.

Finally, the gradient descent method can be applied to the MNIST handwriting recognition problem:
```python
  for x, y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
```

The parameters are re-initialized:
```python
  weights = init_params((28*28, 1))
  bias = init_params(1)
```

A DataLoader is created:
```python
  dl = DataLoader(dset, batch_size = 256)
  xb, xy = first(dl)
```

A validation DataLoader is also created:
```python
  valid_dl = DataLoader(valid_dset, batch_size = 256)
```

One epoch of training via stochastic gradient descent is implemented for this example, in the following way:
```python
  def train_epoch(moedl, lr, params):
    for xb, yb in dl:
      calc_grad(xb, yb, model)
      for p in params:
        p.data -= p.grad*lr
        p.grad.zero_()
```

This function iterates through the data set, calculates the gradient, and updates each of the parameters using SGD before zeroing the gradient for the next step. The `.backward()` method calculates the gradient and adds it to the list of existing gradients. `.zero_()` must be called to clear the list of existing gradients before adding the new one. The `.data()` method informs PyTorch not to update the gradient for that operation.

The distinction between “gradient descent” and “stochastic gradient descent” is that stochastic gradient descent is performed on mini-batches as opposed to working through the algorithm without batches of data. In the example above, the `for xb, yb in dl` line iterates through the mini-batches generated by the `DataLoader`. 

Now, the accuracy of the predictions can be computed by comparing the predictions, `xb`, to the labels, `yb`:
```python
  def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()
```

This can be implemented for the whole validation set in the following way:
```python
  def validate_epoch(model):
    accs = [batch_accuracy(moedl(xb), yb) for xb, yb in vaid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```

In this function, the `.stack()` method stacks the `accs` list items into a tensor, and the `.item()` method converts the mean of that tensor into a scalar with four decimal places, which is indicated by the number after the comma. 

The convergence of the model can be studied by examining the results of iteratively training and validating the model:
```python
  for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end = ‘ ‘)
```

The accuracy increases with each iteration. This is an example stochastic gradient descent optimizer of a linear function.

#### Creating an Optimizer (Simplification Through Refactoring)

The `linear1` function defined previously in this example can be eliminated and replaced by the PyTorch module `nn.Linear`. This is a module that inherits from the `nn.Module` class, defines a linear function of the form `x*w + b`, and initializes its parameters. It is implemented with the following line of code:
```python
  linear_model = nn.Linear(28*28,1)
```

The structure of the weights variable, `w`, defined by calling this method is a vertical vector tensor (`1` column and `28*28` rows), and the bias variable, `b`, is a one-by-one tensor. This can be seen by calling the `.shape()` method on the parameters created by the method:

Running 
```python
  w, b = linear_model.parameters()
  w.shape, b.shape 
```
gives
```
  (torch.Size([1, 784]), torch.Size([1]))
```

This information can be implemented by creating a complete optimizer
```python
  Class BasicOptim:
    def __init__(self, params, lr): self.params, self.lr = list(params), lr
		
    def step(self, *args, **kwargs): 
       for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs): 
      for p in self.params: p.grad = None
```

An instance of this class can be created for the model’s specific parameters”
```python 
  opt = BasicOptim(linear_model.parameters(), lr)
```

The training function can now be simplified from previous versions
```python
  def train_epoch(model):
    for xb, yb in dl:
       calc_grad(xb, yb, model)
       opt.step()
       opt.zero_grad() 
```

Use of the validation function remains the same, and can be implemented to study convergence by defining a model training function: 
```python
  def train_model(model, epochs): 
    for i in range(epochs):
      train_epoch(model)
      print(validate_epoch(model), end=’ ‘)
```

A library function that does exactly what `BasicOptim` does is provided within the *fastai* library. It is called `SGD`. Using this function the equivalent of the code above is the following:
```python
  linear_model = nn.Linear(28*28, 1)
  opt = SGD(linear_model.parameters(), lr)
  train_model(linear_model, 20)
```

The `.fit()` method from the *fastai* library can be used instead of `train_model` function to create a learner. 

`DataLoaders` is a class that stores `DataLaoder` objects with train and validation attributes. 

`Lerarner` is a class with data loader, function, optimization method, loss function, and metric attributes. 

### 2.5 Expanding SGD to Non-linearity and Building a Neural Network <a class="anchor" id="#section_2_5"></a>

#### Nonlinearity

The stochastic gradient descent method discussed to this point can be expanded to encompass optimization on non-linear functions. 

In a very basic way, a neural network can be defined in the following way:
```python
  def simple_net(xb):
    res = xb@w1 +b1
    res = res.max(tensor(0.0))
    res = res@w2 +b2
    return res
```

The neural net consists of two linear functions. All negative values in the first linear function are made to be zero. This is done with the `.res.max(tensor(0.0))` line, which is officially referred to as a rectified linear unit (ReLU). (ReLU is also available as `F.relu` in *PyTorch*.) The second linear function is derived from the first one. 

`w1` and `w2` are weight tensors, and `b1` and `b2` are bias tensors. All of them can be initialized randomly:
```python
  w1 = init_params((28*28, 1))
  b1 = init_params(30)
  w2 = init_params((30,1))
  b2 = init_params(1)
```

Adding the ReLU to the first linear function in the series of two, introduced as a neural network above, creates a situation where the *Univerasl Approximation Theorem* applies. By adding the non-linearity between linear layers, any arbitrary function can theoretically be approximated using this structure given that the weight and bias matrices are large enough.

With a model, that can approximate any function, the limitation shift to resources such as computing power. All further modifications to the framework presented above are for performance optimization.

This model can be simplified using *fastai* library functions:
```python
simple_net = nn.Sequential(
	nn.Linear(28*28, 30),
	nn. ReLU(),
	nn.Linear(30,1)
)
```

As can be seen in both of these examples, the framework of a neural network is function decomposition, where the result of one function is passed to the next, iteratively. In the case of a neural network, function composition is done with linear functions such as `res = xb@w1 +b1` or `nn.Linear(28*28, 30)`, and activation functions or non-linearities such as `res = res.max(tensor(0.0))` or `nn. ReLU()`.

With a ReLU, there is a risk of entire mini-batches becoming zero after the first layer. These are called dead units. There are a few methods to avoid this including choosing sensible initial parameters or a change to a non-zero =-sloped function in the first layer.

The new neural network can be trained using a learner:
```python
learn = Learner(dls, simple_net, opt_func = SGD, loss_func = mnist_loss, metrics = batch_accuracy)
learn.fit(40, 0.1)
```

Generally, neural networks with more layers tend to be less smooth and should be trained with smaller learning rates. 

The convergence of the learner can be studied with the `.recorder` method, for example by plotting accuracy:
```python
plt.plot(L(learn.recorder.values).itemgot(2));
```

This example is simple and effective, but neural networks within the *fastai* library such as the following one with 18 layers, can achieve 99.71% efficiency within the first epoch of training:
```python
dls = ImageDataLoaders.from_folder(pathP
learn = cnn_learner(dls, resnet18, pretrained = False, loss_func = F.cross_entropy, metrics = accuracy)
```

#### Image Classification

In this section, the approach to machine learning shifts back from a conceptual understanding of how they work to practical applications of written library functions. 

Provided datasets can be downloaded using `untar_data` function:
```python
from fastai2.vision.all import *
path = untar_data(URLs.PETS)
```

To display the paths of the data relative to `path` above, where the data is downloaded, use `Path.BASE_PATH = path` before displaying the contents of the directory with `path.ls()`.

In the `PETS` data set, dogs are distinguished from cats via capitalization of the first letter of the file name, and breeds are distinguished by the file names themselves. 

#### Regular Epxpressions

“Regular expressions”, or “regex” can be used to partition strings, such as the names of the files in the `PETS` dataset in useful ways. 
```python
re.findall(r’(.+)_\d+.pjg$’, fname.name)
```

This line of code grabs all contents inside parenthesis without any special treatment for backslashes.

This can be implemented in the construction of a data block that associates independent (data) with dependant (labels) variables and then the creation of a data loader:

```python
pets = DataBlock(
	blocks = (ImageBlock, CategoryBlock),
	splitter = RandomSplitter(seed = 42),
	get_y = using_attr(RegexLabeller(r’(.+)_\d+.pjg$’, fname.name), ‘name’),
	item_tfsm = Resize(460),
	batch_tfsm = aug_transforms(size = 224, min_scale = 0.75))
dls = pets.dataloaders(path/”images”)
```

The combination of the `item_tfsm = Resize(460)` and `batch_tfsm = aug_transforms(size = 224, min_scale = 0.75))` lines above creates more variability in the data augmentation for this model. Complex augmentation done by `aug_transforms` is done in batches on teh GPU. THis method of data augmentation avoids losses. 

#### Debugging Computer Vision Models

Debuggin in a computer vision model is done using the `.show_batch()` method:
```python
dls.show_batch(nrows = 1, ncols = 1)
```

The data augmentations for one specific datapoint (image) can be seen by including `unique = True`:
```python
dls.show_batch(nrows = 1, ncols = 1, unique = True)
```

The `.summary()` is also effective for debugging, particularly when called on a class like `DataBlock`, because it outputs what the script is doing step-by-step and shows where things might be going wrong. Here is an example of its use in practice:
```python
pets1 = DataBlock(
	blocks = (ImageBlock, CategoryBlock),
	get_items = get_image_files,
	splitter = RandomSplitter(seed = 42),
	get_y = using_attr(RegexLabeller(r’(.+)_\d+.pjg$’, 	fname.name), ‘name’),
	pets1.summary(path/”images”)
```

Note that data cleaning does not necessarily need to be done before the model is made. The partially trained model can be used to help clean the model. 

#### Cross-Entropy Loss

*fastai* has the capability of trying to pick a suitable loss function if one is not given. One that is frequently used is “cross-entropy loss”. The previous method of normalizing a probability of two events between `0` and `1` only works for models with binary outcomes. When more categories are included, other loss functions must be used. 

Consider a data batch from the the `.one_batch()` method:
```python
x, y = dls.one_batch()
```

Next, consider the predictions for the mini-batch of 64 items generated above:
```python
preds,_ = learn.get_preds(dl = [(x, y)])
```

The mini bath is passed into `.get_preds` as a data loader to generate a list of predictions. All 37 predictions sum up to $1.0000$ which is expected. 

Softmax can be used to ensure that the predictions for non-binary outcomes sum to one. Effectively, Softmax is an extension of the Sigmoid function that has this capability.

In the current example, we want activations for 37 categories instead of two. For two categories, this is actualized by taking the sigmoid of the relative probabilities of a data point being a specific label. For more than two categories, it is actualized with the Softmax function, which is defined as follows:
```python
def softmax(x): return exp(x) / exp(x).sum(dim = 1, keepim = true)
```

In practice, the *PyTorch* `.softmax()` method can be used directly”

```python
sm_acts = torch.softmax(acts, dim = 1)
```

Softmax is not always the best approach, but it is the default because it works well in many situations.

For the pet classification example, the softmax approach is implemented by returning specific columns for each row of some data, in the following way:
```python
targ = tensor([0, 1, 0, 1, 1, 0])
idx = range(6)
sm_acts[idx, targ]
```

This method is actually identical to the `torch.where(targes == 1, 1 - inputs, inputs).mean()` line used in the binary predictions example.
```python
-sm_acts[idx, targ] 
```

This is applied to the activations matrix, `sm_acts`, for indices, `idx` from zero to one for the targets `targs`.

In practice, the *PyTorch* `.nll_loss()` method:
```python
F.nll_loss(sm_acts, targ, reduction = ‘none’)
```

“nll” stands for “negative log-likelihood”. 

A logarithm can be applied to transform probabilities between `0` and `1` to values between negative infinity and infinity.

Logarithm identities such as $\log\left(a\times b\right) = \log\left(a\right) + \log\left(b\right)$ are particularly useful in machine learning. The right-hand side of this identity is used because it avoids the occurrence of numbers that are too small or too large for the computer to store accurately.

The negative log-likelihood is defined as the mean of logarithms.

Cross entropy loss is defined as the mean log-likelihood of the softmax or the following:
```python
acts.log_softmax().nll_loss
```

Note that the `.nnl_loss()` method does not take the logarithm of the terms that are passed to it. This is because it is computationally more convenient to take the logarithm at the softmax step.

In practice, the *PyTorch* library method `.` can be used:
```python
nn.CrossEntropyLoss(reduction = ‘none’)(acts, targ)
```

Alternatively, `F.cross_entropy()` can be called as a function. It returns a mean, which can be used for a loss function.

#### Training Convolutional Neural Networks for Computer Vision

Cross-entropy loss is an important concept in building neural networks. 

#### Model Interpretation

Confusion matrices are a useful tool for interpreting a model’s effectiveness. However, the `.most_confused()` method can be used to identify which combinations of data and labels the model got most confused on:
```python
interp.most_cpmfused(min_val = 5)
```

#### Model Improvement

There are several methods for model improvement. One is improving the learning rate. This method can be implemented by calling the `.fine_tune()` method with a higher learning rate:

```python
learn = cnn_learner(dls, resnet34, metrics = error_rate)
learn.fine_tune(1, base_lr = 0.1)
```

In some cases, an inappropriate learning rate will increase the error rate of the model. One method of finding an effective learning rate is to use Leslie Smith’s *Learning Rate Finder* which progressively increases the learning rate for each batch of data processed. A plot of loss against learning rate allows a learning rate for which loss is minimum (divided by 10) or decreasing fastest to be chosen:

```python
learn = cnn_learner(dls, renset34, metrics = error_rate)
lr_min, lr_steep = learn.lr_find

print(f”Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}”
```

Note that different learning rates are most effective at different stages in the training process. 

The `.fine_tune()` method was designed for transfer learning, where a model is adapted to generate accurate predictions for a specific sub-application. The process of transfer learning retrains (from a re-randomized initialization) the parameters within the final layer of the model’s neural network. 

Within the `.fine_tune()` method, `.freeze()` is called so that all weights except those in the final layer will be kept constant, while the weights in the final layer will continue to have SGD applied to them. So, only the randomly re-initialized weights in the final layer are re-trained on the new data. Then, all parameters are unfrozen and a learning rate half as large is used to train all the weights.

`.lr_find()` and `.fine_tune()` can be applied iteratively to find progressively better learning rates as the model approaches greater accuracies. Through this process, the earlier layers of the neural networks often get trained more and better than later layers. So, it would be ideal to apply a smaller learning rate to the earlier layers which are closer to having ideal weights and larger learning rates to later layers which are further from having their ideal weights. This is possible in *fastai* by passing a `slice` to the learning rate parameter of `.fit_one_cycle()`:
 
```python
learn = cnn_learner(dls, resnet34, metrics = error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max = slice(1e-6, 1e-4))
```

The notation `slice(1e-6, 1e-4)` indicates that the learning rate of the first layer will be $10^{-6}$ and the learning rate of the last layer will be $10^{-4}$. Layers in between will have arithmetically distributed weights. 

`.fit_one_cycle()` starts at a low learning rate (first argument) and progressively increases its maximum learning rate (second argument).

Note that loss and error rates will not always change simultaneously. In some cases, one will change before the other. It’s important to understand that the error rate is more important than the loss. 

If this approach (choosing good learning rates) to model improvement does not yield accurate enough results, an effective second approach is making the architecture deeper by adding more layers to the neural network. The ResNet architecture, pre-trained on Imagenet is available in 18-, 34-, 50-, 101-, and 152-layer variants.

If a model uses too much memory, the `to_fp16()` method can be called to use half as many bits in the GPU computation it runs:
```python
from fastai.callback.fp16 import *
learn = cnn_learner(dls, resnet50, metrics = error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs = 3)
```

Calling this method also often results in two- to three-times faster training.

It is often wise to experiment with model improvement on small models before applying techniques to larger ones because of the variance in computation speeds. 

### 2.6 Other Computer Vision Problems (PASCAL Multi-lable BIWI Regression Computer Vision Examples) <a class="anchor" id="#section_2_6"></a>

More complicated computer vision problems involve assigning multiple labels to single data points (images).

Consider the PASCAL image dataset:
```python
from fastai.vision.all import *
path = untar_data(URLs.PASCAL_2007)
```

The dataset has a CSV which assigns labels and validity to each data point (image). It can be accessed using Pandas:
```python
df = pd.read_csv(path/‘train.csv’)
df.head()
```

#### Pandas and DataFrames

Pandas is a Python Library containing the class `DataFrames` which can store data in rowans and columns indexed similarly to spreadsheets. The following is a list of a few useful properties of `DataFrames`:
* `iloc` is used to access rows and columns with the following syntax:
	* `df.iloc[:, 0]` to display a whole column
	* `df.iloc[0, :]` or `df.iloc[0]` to display a whole row.
	* `df[‘fname’]` to grab the `fname` column.
* A new column can be created with the following syntax:
```python
df1 = pd.DataFrame()
df1[‘a’] = [1, 2, 3, 4]
df1
```
Alternatively, new columns can be added with this syntax:
```python
df1[’b’] = [10, 20, 30, 40]
df1[‘a’] = df1[‘b’]
```

Although they can be unintuitive, Pandas is a useful library, particularly in machine learning applications. The creator of Pandas wrote the book *Python for Data Analysis*, which is a useful resource for the library.

#### DataBlock and DataLoader

A DataBlock and DataLoader are uniquely created for this application.

A `Dataset`, like the one below, is a collection of tuples of independent and dependant variables:
```python
a = list(enumerate(string.ascii_lowercase))
a[0], len(a)
```
This gives
```
	((0, ‘a’), 26)
```
Which shows how `Dataset`s can be indexed and their lengths can be returned. 

Once a `Dataset` has been constructed, it can be passed to a `DataLoader` in the following way
```python
	dl_a = DataLoader(a, batch_size = 8, shuffle = True)
	b = first(dl_a)
	b
```
This gives
```
	(tensor([7, 4, 20, 19, 5, 25, 22, 13]),(‘h’, ‘e’, ‘u’, ‘t’, ‘f’, ‘z’, ‘w’, ‘n’))
```

To see how the independent and dependant variables correspond to one another, the `zip()` function can be used:
```python
	list(zip(b[0], b[1]))
```
This gives
```
[ (tensor(7), ‘h’),
  (tensor(4), ‘e’),
  (tensor(20), ‘u’),
  …]
```

Python has convenient syntax to pass each element of a data structure into a function like `zip()` using an asterisk:
```python
list(zip(*b))
```
This gives the same output as the previous example. 

`Datasets` is an object that has an associated training `Dataset` and validation `Dataset`. The following is an example of one:
```python
a = list(string.ascii_lowercase)
dss = Datasets(a)
```

Functions to compute the independent and dependent variables can be passed into `Datasets` as well in the following way:

```python
def f1(o): o+‘a’
def f2(o): o+‘b’
dss = Datasets(a, [[f1], [f2]]
```

In this example, `f1` is used to compute the independent variable, and `f2` to compute the dependant variable, both with the parameter `a`. However, the syntax `[[f1, f2]]` can be used to apply both `f1` and `f2` to computing the independent variable with the parameter `a`.

With a `Datasets`, a `DataLoaders` can be constructed in the following way:
```python
dls = DataLoaders.from_dsets(dss, batch_size = 4)
```

It is often convenient to use `DataBlock` instead of going through the manual process of constructing `Dataset`s, a `Datasets`, and then a `DataLoaders`. To do this, start with an empty `DataBlock`, and then take the `Datasets` from it with the `.datasets()` method :
```python
dblock = DataBlock()
dsets = dblock.datasets(df)
```

By default, 80% of the dataset has been allocated to training, and the remaining 20% has been set aside for validation. The `Datasets` can be deconstructed:
```python
x, y = dsets.train[0]
x, y
```
This gives
```
(fname			003213.jpg
 lables			person cat
 is_valid		True
 Name: 1620, dtype: object, fname 003213.jpg
 lables			person cat
 is_valid		True
 Name: 1620, dtype: object)
```

For the case of the PASCAL dataset, the labels or dependant variables will be derived from the file names which can be accessed in the following way:
```python
x[‘fname’]
```

This can be implemented using a lambda which allows a function to be defined and used in the same line of code:
```python
dblock = DataBlock(get_x = lambda r: r[‘fname’], get_y = lambda r: r[‘lables’])
dsets = dblock.datasets(df)
dsets.train[0]
```

Note that issues can arise when saving lines that include `lambda`s. So, a safer alternative is the following:

```python
def get_x(r): return r[‘fname’]
def get_y(r): return r[‘lables’]
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```

For the independent variable, `x`, the path is required to be able to open the images, and for the dependant variable, `y`, additional string formatting is required:
```python
Path.BASE_PATH = path

def get_x(r): return path/‘train’/r[‘fname’]
def get_y(r): r[‘lables’].split(‘ ’)
dsets = DataBlock(get_x = get_x, get_y = get_y)
dblock = dblock.datasets(df)
dsets.train[0]
```

`DataBlock` allows the types of data blocks that are required to be defined. In the case of the PASCAL dataset, a `ImageBlog` is required for the independent variable, and a `MultiCategoryBlock` is required for the dependant variable:
```python
dblock = DataBlocks(blocks = (ImageBlock, MultiCategoryBlock), get_x = get_x, get_y = get_y)
dsets = dblock.datasets(df)
dsets.train[0]
```
The independent and dependant variables are returned as follows:
```
PILImage mode=RGB size=500x375, TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,]))
```
Comparisons between activations and categories are done with the `TensorMultiCategory`, which represents which categories the image belongs to (or, has been assigned to it, in other words). The `.vocab()` method can be used to find out which categories the various elements of the `TensorMultiCategory` vector are associated to:
```python
idxs = torch.where(dsets.train[0][1] == 1.)[0]
dsets.train.vocab[idxs]
```

By default, `DataBlock` uses a random split; however, the CSV for the PASCAL data set includes a column indicating which data points should be used for validation. A splitter can be defined to separate these data points from the rest of the data set:
```python
def splitter(df):
	train = df.index[~df[is_valid]].tolist()
	valid = df.index[df[‘is_valid]].tolist()
	
dblock = DataBlock(blocks = (ImageBlock, MultiCategoryBlock),  splitter = splitter, get_x = get_x, get_y = get_y)

dsets = dblock.datasets(df)
dsets.train[0]
```

The last piece of data management that needs to be added before training are augmentations:
```python
dblock = DataBlock(blocks = (ImageBlock, MultiCategryBlock), splitter = splitter, get_x = get_x, get_y = get_y, item_tfsm = RandomResizedCrop(128, min_scale = 0.35))

dls = dblock.dataloaders(df)
```

#### Binary Cross Entropy Loss Function

Before assigning a loss function, a learner is created:
```python
learn = cnn_learner(dls, resnet18)
```

The shape of the activations in the final layer of the model are studied by deconstruction the mini batches into independant (`x`) and dependant (`y`) variables, and applying the `.size()` method to the activations matrix:
```python
x, y = dls.train.one_batch()
activs = learn.model(x)
activs.shape
```
This gives
```
torch.Size([64, 20])
```
This shape is correct for a mini-batch of 64 items that all have the possibility of being in 20 categories.

The current activations in the last layer of the `resnet18` model can be studied:
```python
	\activs[0]
```

These activations are random at this point, because `resnet18` is a pre-trained model that will have the activations of its last layer retrained for a new task. They are also not between zeor and one, so the `.sigmoid()`, `.log()`, and `.mean()` methods are applied to compute the binary cross entropy loss:
```python
def binary_cross_entropy(inputs, targets):
	inputs = inputs.sigmoid()
	return torch.where(targets == 1, 1- inputs, inputs).log().mean()
```
Because of broadcasting, the last line of the code above applies the “log mean” to each item in the targets matrix.

*PyTorch* includes library functions which are analogous to the `binary_cross_entropy` function defined above. For the following module and function, the initial Sigmoid is not included:
* `F.binary_cross_entropy`
* `nn.BCELoss`
The initial Sigmoid is included in the following module and function:
* `F.binary_cross_entropy_with_logits`
* `nn.BCEWithLogitsLoss`

The equivalents for image classification where categorization can be described with a boolean data type (ie. if a data point is not one category, it is the other), are the following modules and functions:
Without Softmax,
* `F.nll_loss`
* `nn.NLLLoss`
With Softmax,
* `F.cross_entropy`
* `nn.CrossEntropyLoss`

For the PASCAL dataset, loss implementation has the following form, since there a is one-hot encoded target:
```python
	loss_func = nn.BCEWithLogitsLoss()
	loss = loss_func(activs, y)
	loss
```

The loss is computed by comparing activations to targets.

With loss defined, a metric should also be defined. Since accuracy defined as one minus the loss only works for datasets with single labels, a new accuracy metric must be defined:
```python
def accuracy_multi(inp, targ, thresh = 0.5, sigmoid = True):
	“Compute accuracy when `inp` and `targ` are the same size.”
	if sigmoid: inp = inp.sigmoid()
	return ((inp>thresh) == targ.bool()).float().mean()
```
Unlike the “one minus loss” definition which “compute[s] accuracy with `targ` when `pred` is `bs * n_classes`”, the new definition “compute[s] accuracy when `inp` and `targ` are the same size”. In the new definition, different thresholds can be defined as the probability above which a data point is part of a certain category. To use a different threshold, other than the default of 0.5, a partial is used. Partials allow a new function to be created based on a previous function, by fixing one of its parameters. Consider the following example of a partial:
```python
def say_hello(name, say_what = “Hello”): return f”{say_what} {name}.”
say_hello(‘Jeremy’), say_hello(‘Jeremy’, ‘Ahoy!’)

f  = partial(say_hello, say_what = “Bonjour”)
f (“Jeremy”), f(“Sylvain”)
```

If no loss function is defined when a learner is created, an appropriate one will automatically be assigned; however, the threshold which is assigned may not be the best choice. To find the best threshold, accuracy can be plotted against the threshold, and a desirable threshold can be selected through visual analysis:
```python
preds, targs = learn.get_preds()
accuracy_multi(preds, targs, thres = 0.9, sigmoid = False)

xs = torch.linspace(0.05, 0.95, 29)
accs = [accuracy_multi(preds, targs, thresh = i, sigmoid = False) for i in xs]
plt.plot(xs, accs);
```

A concern in choosing parameters in this way is that choosing a hyperparameter (the threshold) might cause the model to overfit. This is not a concern in the current case, because of how smooth and predictable the accuracy/threshold curve is.

#### Image Regression

The distinction between categorization and regression is that categorization refers to the selection or assignment of discrete labels, while regression refers to the assignment of a continuous number. 

One example of image regression is finding the midpoint of a person’s face using the *Biwi* Kinetic Head Pose Dataset to find the midpoint of a person’s head. 

The dataset is accessed in the usual way, using `.untar_data()`:
```python
path = untar_data(URLs.BIWI_HEAD_POSE)
Path.BASE_PATH = path
```

This dataset is organized into 24 directories that correspond to different people. Inside each of the directories are many `_pose` and `_rgb` files that correspond with one another. The dataset is parsed and the contents of the `_pose` file is used to assign a label to its corresponding `_rgb` file. 
```python
img_files = get_image_files(path)

def img2pose(x): return Path(f’{str(x)[:-7]}pose.txt’)

img2pose(img_files[0])
	
im = PILImage.create(img_files[0])
im.shape

im.to_thumb(160)
```

The function provided in the `ReadMe` from the dataset’s website for finding the centre of a person’s head is the following:
```python
cal = np.genformtxt(path/’01’/’rgb.cal’, skip_footer = 6)
def get_ctr(f):
	ctr = np.genformtxt(img2pose(f), skip_header = 6)
	c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
	c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
	return tensor([c1, c2])
```
This function returns the coordinates of the centre of a person's head, and is called in the following way:
```python
get_ctr(image_files[0])
```

Now, a `DataBlock` can be defined. The format of the dependant variable is a `PointBlock` which allows coordinates to be stored:
```python
biwi = DataBlock(
	blocks = (ImageBlock, PointBlock),
	get_items = get_image_files,
	get_y = get_ctrs,
	splitter = FuncSplitter(labda o: o.present.name == ‘13’),
	batch_tfms = [*aug_transforms(size = (240, 320)), 		Normalize.from_stats(*imagenet_stats)]
)
```

With the `DataBlock` and `path`, a `DataLoaders` can be defined:
```python
dls = biwi.dataloaders(path)
dls.show_batch(max_n = 9, figsize = (8, 6))
```

In addition to studying the data, it’s format should be analyzed as well:
```python
xb, yb = dls.one_batch()
xb.shape, yb.shape
```

Shapes of the independent- and dependent-variable mini-batches should be what was expected. In this case, the independent variable has an added dimension with three layers representing the red, green, and blue channels of the image. 

For regression, it is important to define a range within which predictions can be generated. In this case, coordinates are normalized to be between `-1` and `1`. So, the following learner definition with the addition of `y_range` is appropriate:
```python
learn = cnn_learner(dls, resnet18, y_range = (-1, 1))
```

This normalization is accomplished using a Sigmoid function in the following way:
```python
def sigmoid_range(x, lo, hi)L return torch.sigmoid(x) * (hi-lo) + lo
```

Since no loss function was defined in the declaration of the learner, `MSELoss` (mean-squared error) is chosen by default.

No metrics were defined either, but mean-squared error works well. 

Now a learning rate can be defined and fine-tuning on the `resnet18` model can be done:
```python
lr = 1e-2
learn.fine_tune(3, lr)
```

The ability of a model pre-trained on image classification, and re-trained on image regression, to generate accurate predictions is surprising. This is attributed to the nature of the data. In both cases, images are used. 

### 2.7 Tabular Machine Learning with Collaborative Filtering (MovieLens Collaborative Filtering Example) <a class="anchor" id="#section_2_7"></a>

Collaborative filtering allows recommendations to be made based on similarities between two users. For example, movie recommendations are made based on overlapping interests between users. Recommendations of this type are called “latent factors.” Collaborative filtering has applications whenever historical behaviour should be projected forward to predict future behavior.

Consider the dataset, MovieLens, which contains movie rankings on the order of magnitude of $10^{7}$ (tens-of-millions). The data is collated with user, movie, rating, and timestamp columns. 
```python
from fastai2.collab import*
from fastai2.tabular.all import *
	
path = untar_data(URLs.ML_100k)

ratings = pd.read_csv(path/’u.data’, delimiter = ‘\t’, header = None, names = [‘user’, ‘movie’, ‘rating’, ‘timestamp’])
ratings.head()
```

Another method of looking at the dataset is to tabulate scores in a table of users as rows, and movies they’ve seen as columns:

The objective of collaborative filtering in this MovieLens example is to create unique recommendations for viewers about movies they have not yet watched. This can be done with correlations between movies, such as what genre they fall into and share with other movies. Alternatively, this can be done with correlations that exist based on what movies each user watched. Based on the assumption that if two users like the same movies, they might like movies that the other user has seen, this can be effective.

Consider an array of rankings for a movie called “last skywalker” which represent science fiction, action, and old movies:
```python
last_skywalker = np.array([0.98, 0.9,-0.9])
```

Next, consider an array which represents a user’s preferences for the same genres:
```python
user1 = np. array([0.9, 0.8, -0.6])
```

A number that quantifies the match between a users preference and a move can then be calculated as the dot product of the two arrays:
```python
(user1 * last_skywalker).sum()
```
The values in these two arrays used to rank known or unknown categories are the “latent values”

Unfortunately, the latent factors are unknown. But, they can be learned through machine learning.

#### Learning Latent Factors

Values are calculated as the dot product of latent factors for a movie and a user. These are then compared to real data about ratings users gave movies and optimized by changing their latent values.

Before proceeding a table of movie names is joined to the ratings table using the `.merge()` method from Pandas:
```python
movies = pd.read_csv(path/’u.item’, delimiter = ‘|’, encoding = ‘latin-1’, usecols = (0, 1), names = (‘movie’, ‘title’), header = None)
ratings = ratings.merge(movies)
ratings.head()
```

A `DataLoaders` for this collaborative filtering application is created:
```python
dls = CollabDataLoaders.from_df(ratings, item_name = ‘title’, bs = 64)
dls.show_batch()
```
By default, the `ratings` are titled “ratings”, but the `item_name`s are not, so “title” is selected, and the batch size `bs` is set to 64.

Note that in the MovieLens dataset, each movie and its data is stored as an instance of a class. The same applies for users. 

A matrix for collaborative filtering must be initialized with random entries:
```python
n_users = len(dls.classes[‘user’])
n_movies = len(dls.classes[‘title’])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)
```

The match result for a given user and movie is calculated as the dot product of the row/column of latent values that corresponds with the user (from `user_factors`) and the movie (from `movie_factors`).

This is done by matrix multiplying a `one_hot` encoded vector corresponding to the desired index by the `user_factors` and `movie_vectors`:
```python	
one_hot_3 = one_hot(3, n_users).float()
user_factors.t() @ one_hot_3
```
Unfortunately, this is a relatively inefficient way to do matrix look-up. The process of “matrix embedding” makes this process less computationally demanding. Matrix embedding has the same computational speed as array lookup and the same gradients as matrix multiplication.

Instead of using a one-hot encoded matrix, an array lookup is used:
```python
user_factors[3]
```

With this information, a complete collaborative filtering model can be constructed.
Data structures such as classes are used to do this:
```python
class Example:
	def __inti__(self, a): self.a = a
	def say(self, x): return f’Hello {self.a}, {x}.’	
```
Notice that for the definition of each method within the class, `self` is declared as a parameter.

An “instance” of this class can now be created:
```python
ex = Example(‘Sylvain’)
ex.say(‘nice to meet you’)
```
This gives
```
‘Hello Sylvain, nice to meet you.’
```

Definition of classes also allows inheritance when a parameter is included in the class definition. For this example of collaborative filtering, all of the functionality of `Module` is included in `DotProduct` by passing it in during the definition of the class. Additionally, the `Embedding` function, previously discussed, is implemented in the definition of the `DotProduct` class:
```python
class DotProduct(Module):
	def __init__(self, n_users, n_movies, n_factors):
		self.user_factors = Embedding(n_users, n_factors)
		self.movie_factors = Embedding(n_movies, n_factors)

	def forward(self, x):
		users = self.user_factors(x[:, 0])
		movies = self.movie_factors(x[:, 1])
		return (users * movies).sum(dim = 1)		
```

Any time a method inherited from `Module` is called, `.forward()` from `DotProduct` will be called as well. `x` being passed into `.forward()` is a matrix with two columns, the $0^{\text{th}}$ one of which contains the indices for user factors, and the $1^{\text{th}}$ or which contains the indices for the movie factors. The dot product is summed over the compatibility dimension(1), not the mini-batch dimension (0).

Assessing the size of the independent variable of the `DataLoaders reveals a matrix of 64 items each of which has a movie and a user item:
```python
x, y = dls.one_batch()
x.shape
``` 
This gives
```
torch.Size([64, 2])
```

Now a model and learner can be defined:
```python
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func = MSELossFlat())
```

Lastly, the model can be fit:
```python
learn.fit_one_cycle(5, 5e-3)
```

To improve results, a regression range should be defined using `y_range`:
```python
class DotProduct(Module):
	def __init__(self, n_users, n_movies, n_factors, y_range = (0, 5.5)):
		self.user_factors = Embedding(n_users, n_factors)
		self.movie_factors = Embedding(n_movies, n_factors)
		self.y_range = y_range

	def forward(self, x):
		users = self.user_factors(x[:, 0])
		movies = self.movie_factors(x[:, 1])
		return sigmoid_range((users * movies).sum(dim = 1), *self*y_range)	
```

Results improve a small amount.

To improve results further, bias (how well or poorly a user usually rates movies) should be accounted for by adding another latent value:
```python
class DotProduct(Module):
	def __init__(self, n_users, n_movies, n_factors, y_range = (0, 5.5)):
		self.user_factors = Embedding(n_users, n_factors)
		self.user_bias = Embedding(n_users, 1)
		self.movie_factors = Embedding(n_movies, n_factors)
		self.y_range = y_range

	def forward(self, x):
		users = self.user_factors(x[:, 0])
		movies = self.movie_factors(x[:, 1])
		res = (users * movies).sum(dim =1, keepim = True)
		res += self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
		return sigmoid_range(res, *self*y_range)	
```

The addition of this latent parameter causes the model to overfit and the loss gets larger.

#### Regularization

Regularization is used as a collaborative filtering analog of data augmentation. It penalizes models with many parameters and training epochs for overfitting. Regularization is also said to “reduce the capacity” of a model. Simply reducing the number of parameters in a collaborative filtering model is not enough. This tends to simplify the shape of the fit model.

The first method of regularization studied is “weight decay” or “L2 linearization”. In weight decay, the size of the parameters is encouraged to be small. This is done by adding the sum of all the weights squared to the loss function. This prevents overfitting because it limits the weights from growing too much and creates an overfit, jagged solution. Weight decay `wd` is a parameter that controls the sum of the squared weights added to the loss:
```python
loss_with_wd = loss + wd * (weight**2).sum()
```

Note that the gradient of this function will be taken for which the weights will have a linear effect:
```python
weight.grad += wd * 2 * weight
```

The results of implementing this in the following way will be an increase in training loss, but a decrease in validation loss, indicating that the model is no longer over-fitting:
```python
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dlsm model, loss_func = MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd = 0.1)
```

#### Embedding Module

To better understand the “embedding” method of indexing into an array, a class with a .embedding()` method is written:
```python
class T(Module):
	def __init__(self): self.a = torch.ones(3)

L(T().parameters())
```

To tell the compiler that the inherited tensor is a parameter to be learned, it must be wrapped in the `nn.Parameter` class, which automatically also calls `.requires_grad_()` when it is called:
```python
class T(Module):
	def __init__(self): self.a = nn.Parameter(torch.ones(3))

L(T().parameters())
```
Note that *PyTorch* automatically uses `nn.Parameter()` for `nn.Linear()`:
```python
class T(Module):
	def __init__(self): self.a = nn.Linear(1, 3, bias = False)

t = T()
L(t.parameters())
```

Generally when a tensor parameter is initialized it should be assigned random values:
```python
def create_params(size):
	return nn.Parameter(torch.zeros(*size).normal_90, 0.01))
```

This can be implemented in the definition of the `DotProductBias` class without the use of `Embedding`:
```python
class DotProduct(Module):
	def __init__(self, n_users, n_movies, n_factors, y_range = (0, 5.5)):
		self.user_factors = create_params([n_users, n_factors])
		self.user_bias = create_params([n_users])
		self.movie_factors = create_params([n_movies, n_factors])
		self.movie_bias = create_params([n_users])
		self.y_range = y_range

	def forward(self, x):
		users = self.user_factors(x[:, 0])
		movies = self.movie_factors(x[:, 1])
		res = (users * movies).sum(dim =1)
		res += self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
		return sigmoid_range(res, *self*y_range)	
```

#### Collaborative Filtering Analysis: Interpreting Embeddings and Biases
The trained movie biases can be seen and interpreted by displaying the movies with the lowest values in the bias vector:
```python
movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes[‘title’][i] for i in dixs]
```
This gives
```
[‘Children of the Corn: The Gathering  (1996)’,
 ‘Lawnmower Man 2: Beyond Cyberspace (1996)’,
 ‘Beautician and the Beast, The (1997)’,
 ‘Crow: City of Angels, The (1996)’,
 ‘Home Alone 3 (1997)’]
```
Accounting for all latent factors, these are the movies that people liked a lot less than they expected they would.

Similarly, the top five movies by bias can be displayed:
```python
idxs = movie_bias.argsort(descending = True)[:5]
[dls.classes[‘title’][i] for i in dixs]
```
This gives
```
[‘L.A. Confidential  (1997)’,
 ‘Titanic (1997)’,
 ‘Silence of the Lambs, The (1991)’,
 ‘Shawshank Redemption (1994)’,
 ‘Star Wars (1977)’]
```
Accounting for all latent factors, these are the movies that people liked a lot more than they expected they would.

In addition to visualizing and interpreting the biases, the factors can be displayed for interpretation using the `.pca()` method for principle components analysis:
```python
g = ratings.groupby(‘title’)[‘rating’].count()
top_movies = g.sort_values(ascending = False).index.values[:1000]
top_idx = tensor([learn.dls.classes[‘title’].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0, fac1, fac2 = movie_pca.t()
idxs = np.random.hoice(len(top_movies), 50, replace = False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figuresize = (12, 12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
	plt.text(x, y, i, colour = np.random.rand(3)*0.7, forntsize = 11)
plt.show()
```
Looking at the resulting plot, it is obvious that the movies are spaced out across a space of varying latent factors according to genre.

 #### *fastai* Library Funcions and Methods (`fastai.collab`)

*fastai* has built-in methods and functions to creat and train collaborative learning models. They are simple and easy to use:
```python
learn = collab_learrner(dls, n_factors = 50, y_range = (0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd = 0.1)
```

The names of the layers of the model can be printed:
```python
learn.model
```
```
EmbeddingDotBias(
	(u_weight): Embedding(944, 50)
	(i_weight): Embedding(1635, 50)
	(u_bias): Embedding(944, 1)
	(i_bias): Embedding(1635, 1)
```

The same collaborative filtering that was previously used can be used for this model generated with the *fastai* library.

#### Embedding Distance and Cosine Similarity

Embedding distance is the distance between one movie and all other movies. Cosine similarity is the angle between two movies. Both are useful metrics for analysing the data (in this case comparing movies):
```python
movie_factors = learn.model.i_weight.weight
idx = dls.classes[‘title’].02i[‘Silence of the Lambs, The (1991)’]
distances = nn.CosineSimilarity(dim = 1)(movie_factors, movie_factors[idx][none])
idx = distances.argsort(descending = True)[1]
dls.classes[‘title’][idx]
```
This gives
```
'Dial M for Murder (1954)'
```

#### Alternate Methods for Collaborative Filtering

Methods, besides the dot product, exist for collaborative filtering:
```python
class CollabNN(Module):
	def __init__(self, user_sz, item_sz, y_range = (0, 5.5), n_act = 100):
		self.user_factors = Embedding(*user_sz)
		self.item_factors = Embedding(*item_sz)
		self.layers = nn.Sequential(user_sz[1] + item_sz[1], n_act), nn.ReLU(), nn.Linear(n_act, 1))
		self.y_range = y_range

	def forward(self, x):
		embs = self.user_factors(x[:,0]), se;f.item_factors(x[:, 1])
		x = self.layers(torch.cat(embs, dim = 1))
		return sigmoid_range(x, *self.y_range)
```

This method is similar because there is a set of user and item factors that are looked up and multiplied to generate a compatibility score. However, instead of computing a dot product, they are concatenated so that they are next to one another, and then they are passed through a small neural network. 

In the first linear layer, the number of inputs is equal to `user_sz[1] + item_sz[1]`, and the number of outputs is equal to `n_act`. A non-linear, rectified linear unit layer is applied, and the final linear layer takes `n_act` inputs and outputs one prediction.

The `CollabNN` class used to create a model, and `Learner` is used to train it:
```python
model = CollabNN(dls, model, loss_func = MSELossFlat())

learn = Learner(dls, model, loss_func = MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd = 0.01)
```

One benefit of this method is that a different embedding size can be used for each instance where it is used. This could not be done with the dot product method. 
```python
embs = get_embs_sz(lds)
embs
```
If `get_emb_sz` is called and a `DataLoaders`, `dls`, is passed in, an appropriate embedding matrix size will be suggested and used. 

This can be implemented to create a model:
```python
model = CollabNN(*embs)

learn = Learner(dls, model, loss_func = MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd = 0.01)
```
By passing in `embs` with the astriks prefix,  the `user_sz`, `item_sz` attributes that are used in `Embedding`.

Comparing these two architectures, it is not immediately evident which will perform better. 

The first alternate collaborative learning architecture:
```python
learn = Learner(dls, model, loss_func = MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd = 0.1)
```

The second alternate collaborative learning architecture: (This is the one with concatenation and linear layers.)
```python
learn = collab_learner(dls, use_nn = True, y_range = (0, 5.5), layers = [100, 50])
learn.fit_one_cycle(5, 5e-3, wd = 0.1)
```

`collab_learner` returns an object of type `EmbeddingNN`:
```python
@delegates(TabularModel)
class EmbeddingNN(TabularModel):
	def __init__(self, emb_szs, layers, **kwargs):
		super().__init__(emb_szs, layers = layers, n_cont = 0, out_sz = 1, **kwargs)
```

### 2.8 Tabular Statistical Models: Random Trees to Collaborative Filtering (*Kaggle Bulldozer Tabular Data Example*) <a class="anchor" id="#section_2_8"></a>

The general applications of `Embeddings` for any type of categorical modelling are studied. Because `Embedding`s just index into an array, they can be used for any discrete categorical data. The number of discrete levels of a variable gas is called its cardinality.

The Rossman sales competition run on Kaggle is studied as an example:

“Competitors were given a wide range of information about various stores in Germany, and tasked with trying to predict sales on a number of days.”

One of the gold medalists used a tabular deep-learning model. All contestants that used deep learning, used far less feature engineering and far less domain expertise. A paper titled *Entity Embeddings for Categorical Variables* was written on this. The article describes the process of embedding by concatenating one-hot encoded vectors and processing them through a neural network. Results are analyzed by plotting cities on a latent value plane. Cities that were proximate on this plane were geographically proximate as well. A very strong correlation between embedding-space and geographical distance was noticed, even though no geographical data was used to train the model. Similar phenomena were noticed for days of the week and months of the year, where elements that are next to each other appeared next to each other in the embedding space.

Embedding is powerful in capturing accurate information about the world through machine learning. 

Google's recommended method for Google Play, described in *Wide & Deep Learning for Recommender Systems* uses a similar method.

Embedding can be expanded to apply to systems with an arbitrary number of categorical and continuous variables.  A tabular data substitute for deep learning is an “ensemble of decision trees” (i.e. Random Forests and Gradient Boosting Machines). Although deep learning is often superior, particularly for images, audio, text, both approaches yield similar results for tabular data.

#### Ensembles of Decision Trees

Decision tree ensembles currently provide faster and easiest ways of interpreting a model. They also require less hyper-parameter tuning and thus yield good results earlier.

This approach generally does not perform well if very-high-cardinality categorical or neural-network-suited data (such as plain text, audio, or visual) is included.

*PyTorch* is not the best Python library to use for decision tree ensembles since it is designed for gradient-based methods. So, scikit-learn `sklearn` is used instead. The book, *Python for Data Analysis* is a useful resource for this library.

Consider the *Blue Book for Bulldozers* dataset from one of the Kaggle competitions.

 To download a Kaggle dataset, the following procedure must be followed:

{% include alert.html text="Add a note about the download procedure for Kaggle competition datasets here." %}

The main CSV in this database is `train.csv`. It includes in rows for each sale, 
* `SalesID`: a unique identifier for the sale
* `MachineID`: a unique identifier for a machine (which can be sold multiple times).
* `saleprice`: what the machine sold for at auction.
*` saledate`: the date of the sale.

Pandas can be used to read into the CSV and look at the columns:
```python
df = pd.read_csv(path/‘TrainAndValid.csv’, low_memory = False)
df.columns
```

Since the columns are difficult to interpret, the model will be used for data-understanding and -cleanup.

Ordinal columns are those that contain discrete elements that have some natural order such as “small”, “medium”, or “large”.

By looking at the documentation for the dataset, the dependent variable can be found. It is the logarithm of `SalePrice`:
```python
dep_var = ‘SalePrice’
df[dep_var] = np.log(df[dep_var])
```

A decision-tree ensemble requires decision trees, which consist of layers of binary questions. For this example the questions and order of the questions required to predict sales price is unknown. The procedure of determining this is relatively simple:
1. “ Loop through each column of the dataset in turn
2. For each column, loop through each possible level of that column in turn
3. Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of the categorical variable)
4. FInd the average sales price for each of those two groups, and see how close that is to the actual sales price of each of the items of equipment in that group, That is, treat this as a very simple “model” where our predictions are simply the average sales price of the item’s group
5. After looping through all of the columns and possible levels for each, ... the plot point which gave the best predictions using [the] very simple model.
6. [now there are] two different groups for [the] data, based on this selected split. Treat each of these as separate datasets, and find the best split for each, by going back to step one for each group
7. Continue this process recursively, … until [some stopping criteria for each group has been reached]—for instance, stop splitting a group further when it has only 20 items in it.”

Essentially, the error rate is used to find the best splitting criteria for each binary decision. 

Before the decision tree can be created, a few data-procession features must be configured:
```python
df = add_datepart(df, ‘salesdate’)
	
df_test = pd.read_csv(path/‘Test.csv’, low_memory = False)
df_test = add_datepart(df_test, ‘salesdate)
```

This adds a lot of date-related columns to the dataset, which can be seen by calling the following line of code:
```python
‘ ‘.join(0 for o in df.columns if o.startswith(‘sale’))
```

This is a good example of “feature engineering” where lots of potentially helpful data is created for each data point.

#### TabularPandas and TabularProc

The `TabularPandas` class is useful in data cleaning because it allows a Pandas dataset to be created with the provided CSV. Two tabular processors (`TabularProcs`) are used to transform the data. This is a powerful method because it transforms the data in place and does it all at once instead of every time the data is accessed.

`Categorify` replaces a column with a numeric categorical column. `FillMissing` is a `TabularProc` that fills in missing data with the median and appends a new boolean column which is set to true if there was data missing. 

For the Kaggel dataset, the validation set should be the last couple of weeks of data, because the model will be expected to project forward into the future:
```python
cond = df.saleYear < 2011) | (df.saleMonth < 10)
train _idx = np. where(cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx), list(valid_idx))
``` 

When pasing parameters into a `TabularPandas`, a datframe (`df`), tabular procs, and specification about categorical and continuous variables, the dependant variable, and how to split the dataset is required:
```python
cont, cat = cont_cat_split(df, 1, dep_var = dep_var)
to = TabularPandas(df, procs, cat, cont, y_names = dep_var, splits = splits
```

`TabularPandas` behave like *fastai* `Datasets` including the allocation of `.train()` and `.valid()` attributes. The first three rows of the data block can be seen using the following lines of code:
```python
to1 = TabularPandas(df, procs, [‘state’, ‘ProductGroup’, ‘Drive_System’, ‘Enclosure’], [], y_names = dep_var, splits = splits)
to1.show(3)
```

This will return a table populated with labels for categorical data. The analogous numerical values, used during computation can be shown by calling the `.items()` method:
```python
to.items.head(3)
to1.items[[‘state’, ‘ProductGroup’, ‘Drive_System’, ‘Enclosure’]].head(3)
```

The data is now organized in a format from which it can be used to train.

The labels assigned to the values in a column of categorical data can be displayed with the `.classes()` method:
```python
to.classes[‘ProductSize’]
```
This gives, for the current example,
```
(#7)[‘#na#’, ‘Large’, ‘Large / Medium’, ‘Medium’, ‘Small’, ‘Mini’, ‘Compact’]
```

Since tabular data procession takes time, it should periodically be saved using the `.save()` method:
```python
(path/’to.pkl’).save(to)
```

This version of the decisiontree ensemble model can then be retrieved with the `.load()` method:
```python
to = (path/‘to.pkl’).load()
```

The independent and dependant variables are separated:
```python
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
```

Now that all missing data has been filled in and categoric variables have been made numeric, and the independent and dependent variables have been separated, a decision tree can be constructed:

A decision tree where the dependant variable is continuous is called a “decsion tree regressor”.
```python
m = DecisionTreeRegressor(max_leaf_nodes = 4)
m.fit(xs, y)
```

The tree, in its its current state can be displayed with `draw_tree`:
```python
draw_tree(m, xs, size = 7, leaves_parallel = True, precision = 2)
```
	
For each node, the following data is displayed:
* Label, Logic Argument
* Mean Squared Error
* Number of Samples 
* Value of the Dependent Variable

Notice that for each progressive binary decision, the mean squared error progressively decreases. 

Another way to visualize the data is using the *dtreeviz* library:
```
samp_idx = np.ranom.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var, fontname = ‘DejaVu Sans’, scale = 1.6, label_fotsize = 10, orientation = ‘LR’)
````
 Using this method of data visualization can reveal some errors such as many of the bulldozers having 1000 as the year they were made. This most likely occurred when missing data was being filled in.

Now, a larger decision tree, where no stopping criteria is specified, is built:
```python
m = DecisionTreeRegressor()
m.fit(xs, y);	
```

The following functions are used to assess the final root mean square error value:
```python
def r_mse(pred, y): return round(math.sqrt(((pred - y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)
```
This gives
```
0.0
```

The validation must be checked. A value of `0.3377` reveals that the model has been overfit.

Further investigation can be done by printing the number of leaf-nodes and datapoints:
```python
m.get_n_leaves(), len(xs)
```
This reveals that there are nearly as many leaf nodes as data points.

Different stopping criteria should be used to avoid this issue. One would be to stop splitting if a leaf node has 25 items or less with the criteria `min_samples_leaf = 25`:
```python
m = DecisionTreeRegressor(min_samples_leaf = 25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m(rmse(m, valid_xs, valid_y)
```

This has reduced the number of leaf nodes and improved accuracy in the validation set. 

##### Working with Categorical Variables

The approach to handling categorical variables to this point has been to transform them into numerical values. Binary decisions are made with logic conditions based on these numbers; however, the order of the numbers assigned may not correspond with the order of the categories if there even is one. In general, categories that are next to each other should have corresponding numerical values that are next to each other. For example, it would be useful if  “small”, “medium”, “large” corresponded with 1, 2, 3. So that a numerical inequality for a binary split makes sense for the categories as well. 

Missing ordinal values will be assigned zero when data is filled in.

#### Bagging for Random Forests

Retired Berkeley professor, Leo Breiman, developed the method of “Bagging Predictors” as a way of improving random forests in 1994. The method involves training the models on random subsets of the data called “bootstrap replicas”, and taking the average of predictions for each of these models to form a more accurate final prediction.

The proposed procedure is as follows:
1. “Randomly choose a subject of the rows of [the] data (i.e. “bootstrap replicas of [the] learning set”)
2. Train a model using this subject.
3. Save that model, and then return to step one for a few times
4. This will give ... a number of trained models. To make a prediction, predict using all the models, and then take the average of each of the model’s predictions.”

The accuracy of almost any algorithm can be improved with bagging.

In 2001, Breiman adapted his approach to decision trees specifically by randomly;y choosing a subset of rows and columns for each split in a decision tree. This approach may be the most widely used and most practical approach to tabular machine learning.

#### Creating a Random Forest Regressor

A random forest regressor can be constructed and fitted in the following way:
```python
def rf(xs, y, n_estimators = 40, max_samples = 200_000, max_features = 0.5, min_samples_leafs = 5, **kwargs):
	return RandomForestRegressor(n_jobs = -1, n_estimators = n_estimators, max_samples = max_samples, max_features = max_features, min_samples_leaf = min_samples_leaf, oob_score = True).fit(xs, y)

m = rf(xs, y)

m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
```

This method is very effective.

The *sklearn* documentation includes information about how the number of estimators impacts the accuracy. In general, accuracy increases as trees are added. 

To see the affect of `n_estimators` in practice, the `estimators_` attribute can be used:
```python
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
```

This returns a *numpy* array with the predictions from each individual tree, which can then be averaged across the `0` axis to get a final prediction. 
```python
r_mse(preds.mean(0), valid_y)
```
Another way of interpreting this data is plotting the accuracy agains the number of tress in the model:
```python
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);
```

Continuous improvement is noticeable, but the improvements slowed down. 

#### Out-of-Bag Error

“Out-of-bag error”, among other things, can be used to understand whether issues in training arise as a result of overfitting or inconsistencies between training and validation sets. Library functions are included to compute it:
```python
r_mse(m.oob_prediciton_, y)
```

Out-of-bag error ensures that predictions for each tree is computed with a subset of the data that was not used to train it. It is similar to generating predictions with a validation dataset, but there is no time offset. Data that was not included in training is used but it is selected randomly, independent of time.

#### Model Interpretation

Model interpretation is particularly important for tabular data. Howard and Gugger point out several considerations for tabular data:
* How confidence-inspiring are the predictions from a particular row of data?
* How do various factors influence predictions for a particular row?
* What is the predictive power of each column?
* Which columns are highly influential in forming predictions? Which are redundant?
* How do predictions vary with columns?

```python
import warnings
warnings.simplifier(‘ignore’, FutureWarning)

from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall
```

Some predictions are made, and stacked into a *numpy* array:
```python
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
preds.shape
```

The **standard deviation** of the predictions in the `preds` array indicates how much they carried:
```python
preds_std = preds.std(0)
```
Analyzing the standard deviations can flag issues if one stands out from the others. 

**Feature importance** ranks the features in terms of which are most influential in the decision tree:
```python
fi = rf_feat_importance(m, xs)
```
They can also be plotted:
```python
def plot_fi(fi):
	return fi.plot(‘cols’, ‘imp’, ‘barh’, figsize = (12, 7), lengend = False)

plot_fi(fi[:30])
```
This is done by the `.feature_imporance_` from *sklearn*. An algorithm works its way down the tree, tracking what criteria were used for the split and how much the model improved. The result is an indication of which parameters are critical in the model and which could potentially be removed:
```python
to_keep = fi[fi.imp > 0.005]
len(to_keep)

xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]

m = rf(xa_imp, y)
```
Even after some columns (the less significant ones) are removed, the accuracy of the model remains quite good. 

Further **redundant features** can be removed. 

Call *fastai*’s `cluster_columns`:
```python
cluster_columns(xs_imp)
```

This feature helps identify pairs or groups of columns that are similar. For these groupings, when one categorical variable is high, so are the others and vice-versa. Some of them could be removed.

The baseline error before removing any of them is generated by the following line:
```python
get_oob(xs_imp)
```

Now the errors are recorded as variables are removed, one-by-one:
```python
{c:get_oob(xs_imp.drop(c, axis = 1)) for c in (‘saleYear’, ‘saleElapsed’, ‘ProductGroupDesc’, ‘findModelDesc’, ‘fiBaseModel’, ‘Hydraulics_Flow’, ‘Grouser_Tracks’, ‘Coupler_Systems’)]
```

No significant decrease in accuracy is observed as these columns are removed. 

**Partial dependancies** between columns may become evident through looking at the histogram for a particular column:
```python
p = valid_xs_final[‘ProductSize’].value_counts(sort = False).plot.barh()
c = to.classes[‘ProductSize’]
plt.yticks(range(len(c)), c);
```
This will show the number of items with the various labels for that columns.

For `YearMade`, an actual histogram is required:
```python
ax = valid_xs_final[‘YearMade’].hist()
```

A partial-dependence plot will show the correlation between the dependant variable and a categorical independent variable:
```python
from sklearn.inspection import plot_partial_dependance 

fig, ax = plt.subplots(fisize = (12, 4))
plot_partial_dependance(m, valid_xs_final, [‘YearmMade’, ‘ProductSize’], grid_resolution = 20, ax = ax);
```

Note that in general, a dependent variable cannot be plotted against an independent variable, because there are several other independent variables that influence it. However, this approach attempts to plot the influence of just one column on the dependent variable by calculating the average dependent variable for all scenarios if the independent variable were a constant. This is done recursively for all levels of the independent categorical variable. This isolates the effect of the studied independent variable. 

Some issues can arise if some of the data for a given column was not labelled. 

For further interpretation and improvement, the **tree interpreter** libraries are used. First, they must be installed: 
```python
import warnings
warnings.simplifier(‘ignore’, FutureWarning)

from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall
```
```
!pip install treeinterpreter
!pip install waterfallcharts
```

The `treeintepreter` module works by passing in a single row to compute whether the predicted dependant variable increases or decreases with each binary decision. The total importance by a split variable can then be added up:
```python
prediction[0], bias[0], contributions[0].sum()
```
Gives
```
(array([9.98234598]), 10.104309759725059, -0.12196378442186026)
```

Plotting these results:
```python
waterfall(valid_xs_final.columns, contributions[0], threshold = 0.08, roation_value = 45, formatting = ‘{:, 3f}’);
```

This plot will show the impact that each variable has on the dependent variable. It will also show the net effect.

#### Extrapolation and Neural Networks

Begin by creating some synthetic data:
```python
np.random.seed(42)
	
x_lin = torch.linspace(0, 20, steps = 40)
y_lin = x_lin + torch.randn_like(x_lin)
plt.scatter(x_lin, y_lin);
```

A random forest will be used to predict this normally-distributed random data. The horizontal vector is made vertical with the `.unsqueeze()` method:
```python
xs_lin = x_lin.unsqueeze(1)
x_lin.shape, xs_lin.shape
```
An alternate method to achieving this transformation is using `x_lin[:, None].shape`, which places the unit axis where `None` is. 

Now that the data is formatted in the necessary way, random forest can be constructed using the first 30 rows:
```python
m_lin = RandomForestRegressor().fit(xs_lin[:30],y_lin[:30])
```

The predictions can be plotted on the same plane as the origitonal data:
```python
plt.scatter(x_lin, y_lin, 20)
plt.scatter(x_lin, m_lin.predict(xs_lin), color = ‘red’, alpha = 0.5);
```

A flattening of the data for high independent variables is noticeable. Because the data higher than the flat-line value did not appear in the training set. This is a demonstration of the model's poor ability to extrapolate outside the data it has seen. Finding out-of-domain data is an important feature of a model, though.

The sources of the differences between the training and validation training sets can be identified using the following implementation of `rf_feat_importance`:
```python
df_dom = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0]*len(xs_final) + [1]*len(valid_xs_final))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:6]
```

These lines of code detect the predictive power of the model and identify which columns caused the differences between the training and validation datasets.

The columns (at the top of the list) that are very different between the training and validation datasets can be removed to see if the model’s predictive power to extrapolate outside of the training data is increased:
```python
m = rf(xs_final, y)
print(‘orig’, m_rmse(m, valid_xs_final, valid_y))

for c in (‘SalesID’, ‘saleElapsed’, ‘MachineID’):
	m = rf(xs_final.drop(c, axis = 1), y)
	print(c, m_rmse(m, vlaid_xs_final.drop(c, axis = 1), valid_y))
```

Analyzing the model in this way reveals that the leading two columns that account for differences between the training and validation datasets can be removed and the predictive power of the model will increase. This will increase the model’s resilience over time because time dependencies between the training and validation datasets are being removed.

#### Using a Neural Network

Issues with extrapolation do not occur in neural networks:
```python
df_nn = pd.read_csv(path/’TrainAndValid.csv’, low_memory = False)
df_nn[‘ProductSize’] = df_nn[‘ProductSize’].astype(‘catagory’)
df_nn[‘ProductSize’].cat.set_categories(sizes, ordered = True, inplace = True)
df_nn = add_datepart(df_nn, ‘saledate’)
```

The same datafame that was preciously used is used again:
```python
df_nn_final = df_nn[list(xs_final)time.columns) + [dep_car]]
```

For categorical columns, embeddings will be used:
```python
cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card = 9000, dep_var = dep_var)
```
A maximum cardinality (maximum amount of discrete levels), `max_card`, is specified above which the categorical variable is treated as a continuous variable. 

All variables that will be extrapolated are defined as continuous:
```python
cont_nn.append(‘sakeElapsed’)
cat_nn.remove(‘saleElapsed’)
```

The number of uniqu, discrete levels for each categorical variable can be seen using the `.unique()` method:
```python
df_nn_fina[cat_nn].unique()
```
Caution should be used for variables with many discrete levels, because each level will add a row to the embedding matrix. Removing these categorical variables with many discrete levels does not compromise the model’s accuracy significantly; removing one of a set of two variables with similar numbers of discrete levels actually improves the models’ accuracy:
```python
xs_filt2 = xs_filt.drop(‘fiModeDescriptor’, axis = 1)
valid_xs_time2 = valid_xs_time.drop(‘fiModelDescriptor’, axis = 1)
m2 = rf(xs_filt2, y_filt)
m_rmse(m, xs_filt2, y_filt), m_rmse(m2, valid_xs_time2, valid_y)
```
Since the impact is minimal, `fiModelDescriptor` can be removed from the model altogether.
```python
cat_nn.remove(fiModelDescriptor’)
```

A `TabularPandas` object can be created with `Normalize` to make the scale of all variables similar:
```python
procs_nn = [Categotify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits = splits, y_names = dep_var)
```

With a `TabularPandas` object, a `dataloaders` with a batch size of `1024` can be created. A large batch size is chosen because tabular data takes up far less memory than images do. 
```python
dls = to_nn.dataloaders(1024)
```

Since the model is performing regression on a continuous dependant variable, a range for the dependant variable is defined:
```python
y = to_nn.train.y
y.min(), y.max()
```

All parameters are now defined and a tabular model can be created. By default, *fastai* creates a neural network with two hidden layers with 500 and 250 (200 and 100 by default) activation respectively for tabular data:
```python
from fastai2.tabular.all import *

learn = tabular_learner(dls, y_range = (8, 12), layers = [500, 250], n_out = 1, loss_func = F.mse_loss)

learn.lr_find()
```

There are no pre-trained models for this type of tabular data, so the model is trained with 1-cycle for a few epochs:
```python
learn.fit_one_cycle(5, 1e-2)
```

The random mean-squared error, `r_mse`, can be used to compare the results to the forest result achieved earlier:
```python
preds, targs = learn.get_preds()
r_mse(preds, targs)
```
The neural network-based model already has a better result than the tuned random-forest-based one.

#### Ensembling

Both the random-forest approach and the neural-network approach have benefits and drawbacks. Ensembling can be a way of getting the “best of both worlds”. There are many ways to ensemble random-forest and neural-network tabular models, but the simplest (called “bagging” is to take the average of the final prediction of both:
```python
rf_reds = m.predict(valid_xs_time)
ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2

r_mse(ens_preds, valid_y
```
Results are improved by this method.

Another method of ensembling is called “boosting”, which is done by 
* “train[ing] a small model which underfits [the] data,
* calulat[ing] the predictions in the training set of this model,
* subtract[ing] the predictions from the targets (these are called the “residuals”, and represent the error for each point in the training set),
* [return] to step one, but instead of using the original targets, use the residuals as the targets for training,
* continue doing until [some stopping criterion, such as a maximum number of trees, is reached] or [a deterioration in error is observed].”

The predictions of each can be summed (since they are derived from residuals) to form a final prediction.

Other, more complicated variants to this method of ensembling include *Gradient Boosting Machines* (GBMs) or *Gradient-Boosted Decision Trees* (GBDTs) using popular libraries like *XGBoost*. 

What should be noted about bagged and boosted models is that there is no upper limit for fitting. As more trees are added, the model’s accuracy should continue to go up. Gradually the model will start to overfit more and more. Boosted methods require parameter turning. They are quite sensitive to hyper-parameters.

#### Combining Embedding with Other Methods

The entity embedding paper states: “the embeddings obtained from the trained neural network boost the performance of all tested machine learning methods considerably when used as the input features instead.”

A summary of approaches for tabular modelling presented by Howard and Gugger:
* “**Random forests** are the easiest to train, because they are extremely resilient to hyper-parameter choices, and require very little preprocessing. They are very fast to train and should not overfit, if [enough trees are used]. But, they can be less accurate, especially if extrapolation is required, such as prediction future periods.
* **Gradient boosting machines**, in theory, is just as fast to train as random forests, but in practice [many different hyperparameters must be tried]. They can overfit. But they are often a bit more accurate than random forests. 
* **Neural Networks** take the longest time to train, and require extra procession such as normalization.; this normalization needs to be used at inference time as well. They can provide great results, and extrapolate well, but only when [hyperparameters are treated carefully], and [care is taken to avoid overfitting].”

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
