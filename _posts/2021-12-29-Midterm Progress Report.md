# Midterm Progress Report

A report discussing the progress of the *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture* project, halfway through the work period allotted to completing it (4/8 months). 

## Table of Contents

* [1. Introduction](#section_1)
* [2. Project and Progress](#section_2)
  * [2.1 Defining the Scope of the Study](#section_2_1)
  * [2.2 Learning Machine Learning](#section_2_2)
  * [2.3 Applying Machine Learning](#section_2_3)
  * [2.4 Discussing the Results](#section_2_4)
* [3. Gantt Chart Overview](#section_3)
* [4. Next Steps](#section_4)
  * [4.1 Project Scope](#section_4_1)
  * [4.2 Mentorship](#section_4_2)
  * [4.3 Structure](#section_4_3)
* [References](#references)

Version/Revision History:

Version/Revision | Date Published | Details
-----|-----|----- 
V00, Rev.01 | 2021-12-28 | Initial Draft
V01, Rev.00 | 2021-12-29 | Midterm Submission

## 1. Introduction <a class="anchor" id="#section_1"></a>

The objective of this progress report is to provide clarity about the purpose and scope of the project, segment it into definable and actionable items, show the progress that has been made, and discuss what next steps need to be made to ensure successful completion. 

The *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture* project was motivated, somewhat serendipitously, during travel across Canada. The author visited a passive solar home and witnessed the homeowners’ efforts to achieve greater energy savings by opening and closing windows and blinds manually to control the indoor temperature. It became evident that with the assistance of automation and control applications, sustainable living solutions could become more accessible to those prospective homeowners less motivated to pick up new manual house duties. Accordingly, the feasibility of achieving greater energy savings in sustainable residential architecture using automation and machine learning has become the focus of this project.

In subsequent sections, the project is broken up into four large yet definable sections, for each of which the current progress on the project is explained ([Section 2](#section_2)), a Gantt chart overview is provided ([Section 3](#section_3)), and recommendations for improvement during the second half of the work term are made ([Section 4](#section_4)).

## 2. Project and Progress <a class="anchor" id="#section_2"></a>

The *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture* project can be broken down into four major sections: 
* Defining the Scope of the Study
* Learning Machine Learning
* Applying Machine Learning 
* Discussing the Results
Each serves a unique purpose in creating a cohesive, novel, and useful final article.

Progress has been made in the following three areas of the project:
* Defining the Scope
* Labelled Data Obtainment
* Learning Machine Learning
 
Each section of the project, as well as any relevant progress made at this point in the work term, is discussed in succession in the following subsections:

### 2.1 Defining the Scope of the Study <a class="anchor" id="#section_2_1"></a>

The scope of the study being done is driven by the project’s purpose; however, it is also limited by the available time and resources. Defining an appropriate scope is a matter of balancing curiosity and effort. 

A scope that is too broad or too narrow can result in conclusions that are not useful. Too broad a scope can lead to no conclusion or conclusions for which there are too many irrelevant but influential parameters. Likewise, too narrow a scope can lead to conclusions for which there are no or few applications. Bhiladvala [^bhiladvala-21], one of the co-supervisors for this project, stresses the importance of erring on the side of a narrow scope so that the conclusions made are definitive even if their application is limited.

The scope of this study on *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture* was defined early on in the project to ensure that work done throughout the remainder of the term was towards a useful conclusion. 

{% include info.html text="Note: The progress made in this area of the report can be found in the first version of the final article. In the introduction and control system sections, purpose, novelty, and scope are addressed." %}

**Relevant Links:**
* [first version of *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture*](https://simonvincenthk.github.io/2021/12/27/Automation-and-Machine-Learning-for-Grid-Power-Use-Minimization-in-Sustainable-Residential-Architecture.html)

### 2.2 Learning Machine Learning <a class="anchor" id="#section_2_2"></a>

In addition to writing their book, *Deep Learning for Coders with fastai and PyTourch*, Howard and Gugger [^howardandgugger-20] created a set of courses to teach Machine Learning to people who have experience programming with Python. The first course in their series is called [*fastai* “Practical Deep Learning for Coders”](https://course.fast.ai/), and it takes the unconventional approach of beginning with practical applications of the material before finishing with the theoretical background. This is a studied method of teaching which has been proven by the success of this course.

[*fastai* “Practical Deep Learning for Coders”](https://course.fast.ai/) consists of eight lessons, each with a practical assignment:
* Lesson 1 – Your first model
* Lesson 2 – Evidence and p-values
* Lesson 3 – Production and deployment
* Lesson 4 – SGD from scratch
* Lesson 5 – Data ethics
* Lesson 6 – Collaborative filtering
* Lesson 7 – Tabular data
* Lesson 8 – Natural language processing

All of the lessons are relevant to the project and should be completed before finalizing the study so that a broad understanding of the subject matter can be applied.

The lecture and assignment for the first lesson and the lecture for the second lesson have been completed. The remaining lectures and assignments are outstanding at this time. The material learned so far has been included in the current version of the final article as an appendix ([B](https://simonvincenthk.github.io/2021/12/26/Appendix-B-Machine-Learning-with-PyTourch-and-fastai.html)).

{% include info.html text="Note: The progress made in this area of the report can be found in Appendix B. There, machine learning is discussed to give the layperson context and a basic understanding of the applicability and functioning of the technology." %}

**Relevant Links:**
* [Appendix B](https://simonvincenthk.github.io/2021/12/26/Appendix-B-Machine-Learning-with-PyTourch-and-fastai.html)

### 2.3 Applying Machine Learning<a class="anchor" id="#section_2_3"></a>

As a data-driven method, machine learning is highly dependent on both the architecture of the model as well as the data used to train it. Accordingly, several smaller tasks related to architecture and data must be completed to construct a machine-learning-based control application:
* Architecture Selection
* Labelled Dataset Obtainment
* Model Training
* Model Validation
* Model Testing

Architecture selection is conditional on the understanding gained from the previous section, “Learning Machine Learning” as well as a review of common practices in the field of building energy. For an educated decision to be made, an architecture will be selected after gaining an understanding of common practices in building energy and machine learning. 

According to Evins [^evins-21], labelled dataset obtainment is a challenging topic in the field of building energy, so a custom dataset must be generated through computer simulation. The simulation engine used is EnergyPlus, and an understanding of how to construct and run a simulation using the recommended software stack is necessary to generate a quality dada set. 

Howard and Gugger [^howardandgugger-20] explain that model training is subtle, mainly because of the possibility of overtraining or undertraining the model. Experience constructing and training models, gained through completion of the “Practical Deep Learning for Coders”](https://course.fast.ai/) course, will be assets in completing this subsection of the project. 

Model validation is a critical component in understanding the model’s capabilities. [^howardandgugger-20] An understanding of model evaluation will be important in completing this subsection of the project. This will be gained through completing the [*fastai* “Practical Deep Learning for Coders”](https://course.fast.ai/) course.

And, model testing serves as an opportunity to see whether the model performs reliably. [^howardandgugger-20] Similarly to the training and validation subsections, testing the model correctly will depend on the successful completion of the [*fastai* “Practical Deep Learning for Coders”](https://course.fast.ai/) course.

Since the dependencies between relevant parameters and what constitutes a labelled dataset are already understood, progress has been made on creating an EnergyPlus simulation to generate a dataset for training, validation, and testing. All relevant phenomena except buoyancy-driven cooling have been included. Likewise, the simulation runs; however, not all relevant output parameters have been selected. A description of the EenergyPlus simulation is included in the current version of the final article as an appendix ([A](https://simonvincenthk.github.io/2021/12/23/Appendix-A-EnergyPlus-Shoebox-Model.md.html)).

{% include info.html text="Note: The progress made in this area of the report can be found in Appendix A. There, the software used to simulate the situation studied in this project as well as how it is constructed and run is discussed." %}

**Relevant Links:**
* [Appendix A](https://simonvincenthk.github.io/2021/12/23/Appendix-A-EnergyPlus-Shoebox-Model.md.html)

### 2.4 Discussing the Results <a class="anchor" id="#section_2_4"></a>

There are several points of evaluation and discussion for the results:
Whether the machine learning control application can reliably reach the indoor temperature setpoint while minimizing the use of grid power.
Whether there is a notable decrease in energy consumption with automation of windows and blinds and a machine learning control application.
Whether the automation and machine learning approach studied in this report is more broadly applicable, beyond passive solar homes, and sustainable residential architecture. 
Whether there are recommendations for continued improvement. 

Bhiladvala [^bhiladvala-21] particularly encourages study and discussion of the automation and machine learning approach in multiple settings in addition to an explicit statement about the benefits it provides. His motivation in providing this advice is that the final article provides a definitive and useful conclusion.

The discussion and results position of the project has not been addressed at this time. 

{% include alert.html text="Note: Since the study is incomplete, no evaluation or discussion of the results has been done at this time." %}

## 3. Gantt Chart Overview <a class="anchor" id="#section_3"></a>

Based on the discussion of the project and progress provided above, a Gantt chart can be helpful in visualizing which portions have been completed, which are outstanding, and when each of them will be completed:

| Item                                       | Status      | Sep 2021 | Oct 2021 | Nov 2021 |    Dec 2021 | Jan 2022 | Feb 2022 | Mar 2022 |    Apr 2022 |
|--------------------------------------------|-------------|----------|----------|----------|-------------|----------|----------|----------|-------------|
| **Deliverables**                           |             |          |          |          |             |          |          |          |             |
| Midterm Progress Report                    | Complete    |          |          |          | 31 Dec 2021 |          |          |          |             |
| Presentation                               | Incomplete  |          |          |          |             |          |          |          | 30 Apr 2022 |
| Final Report                               | Incomplete  |          |          |          |             |          |          |          | 30 Apr 2022 |
| **Milestones**                             |             |          |          |          |             |          |          |          |             |
|*1. Defining the Scope of the Study*        | Complete    |*Sep 2021*|*Oct 2021*|          |             |          |          |          |             |
| 2. Learning Machine Learning               | In Progress |*Sep 2021*|*Oct 2021*|*Nov 2021*|*Dec 2021*   |*Jan 2022*|          |          |             |
| 2.1 Lesson 1 – Your first model            | Complete    | Sep 2021 | Oct 2021 |          |             |          |          |          |             |
| 2.2 Lesson 2 – Evidence and p-values       | In Progress |          | Oct 2021 |          |             | Jan 2022 |          |          |             |
| 2.3 Lesson 3 – Production and deployment   | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
| 2.4 Lesson 4 – SGD from scratch            | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
| 2.5 Lesson 5 – Data ethics                 | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
| 2.6 Lesson 6 – Collaborative filtering     | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
| 2.7 Lesson 7 – Tabular data                | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
| 2.8 Lesson 8 – Natural language processing | Incomplete  |          |          |          |             | Jan 2022 |          |          |             |
|*3. Applying Machine Learning*              | In Progress |          |*Oct 2021*|*Nov 2021*|*Dec 2021*   |*Jan 2022*|*Feb 2022*|          |             |
| 3.1 Architecture Selection                 | Incomplete  |          |          |          |             |          | Feb 2022 |          |             |
| 3.2 Labeled Dataset Obtainment             | In Progress |          | Oct 2021 | Nov 2021 | Dec 2021    |          |          |          |             |
| 3.3 Model Training                         | Incomplete  |          |          |          |             |          | Feb 2022 |          |             |
| 3.4 Model Validation                       | Incomplete  |          |          |          |             |          | Feb 2022 |          |             |
| 3.5 Model Testing                          | Incomplete  |          |          |          |             |          | Feb 2022 |          |             |
| *4. Discussion of Results*                 | Incomplete  |          |          |          |             |          |          |*Mar 2022*|             |

## 4. Next Steps <a class="anchor" id="#section_4"></a>

The following are three actions that should be taken to ensure the successful completion of this project:
* Narrowing the scope of the project.
* Seeking mentorship from a subject matter expert.
* Establishing a structure for the remainder of the work to be completed. 
Each is discussed in turn in the following subsections:

### 4.1 Project Scope <a class="anchor" id="#section_4_1"></a>

To ensure the successful completion of this project in the allotted work term (September 2021 – April 2022), it may be necessary to narrow the scope of the project. Beyond the limitations provided by the current definition of scope in the introduction and control system sections of the current version of the final article, several phenomena may have to be excluded from the EnergyPlus Simulation and control system. 

Evins [^evins-21] recommended excluding buoyancy-driven cooling because it can be difficult to simulate and because the focus of this study is on the application of automation and machine learning. This simplification should be entertained and discussed with supervisors at the beginning of the second half of the project work term. 

### 4.2 Mentorship <a class="anchor" id="#section_4_2"></a>

Content-specific mentorship may be necessary for the successful completion of this project, specifically in the area of building energy. 

Early in the work term, Dr. Ralph Evnins [^evins-21], a building energy researcher and instructor at the *University of Victoria* was consulted regarding a labelled dataset for this study. He shared some of his recent research, *On the Joint Control of Multiple Building Systems with Reinforcement Learning*, [^zhangetal-21] which is very similar to the work being done on this project, and expressed an interest in getting involved as a supervisor. Together with his ability to define a realistic project scope, Dr. Evins’ technical knowledge of building energy and applications of machine learning would be a great asset to the completion of this project. 

At the beginning of the second half of the work term, Dr. Evnis and the Department of Mechanical Engineering will be consulted regarding his involvement throughout the remainder of the project. 

### 4.3 Structure <a class="anchor" id="#section_4_3"></a>

Successful completion of the project will be achieved by following the structure laid out in the Gantt chart provided in the previous section. To maintain synchronicity with it, 
* two lessons from the [*fastai* “Practical Deep Learning for Coders”](https://course.fast.ai/) course should be completed per week during January (2022),
* the machine learning control application model should be constructed, trained, validated, and tested during February (2022), and 
* all analysis and discussion of the results should be completed during March (2022). 

## References <a class="anchor" id="#references"></a>

[^ardakanianetal-18]: Ardakanian et at., 2018: *Non-intrusive occupancy monitoring for energy conservation in commercial buildings*
[^asa-16]: ASA, 2016: *American Statistical Association Releases Statement on Statistical Significance and P-Values*
[^bigladder-21]: Big-Ladder Software, 2021: *Energy Plus Web-Based Documentation*
[^bhiladvala-21]: Bhiladvala, 2021: *Honors Thesis* 
[^brackneyetal-18]: Brackney et al., 2018: *Building Energy Modeling with OpenStudio*
[^dingetal-19]: Ding et at., 2019: *OCTOPUS: Deep Reinforcement Learning for Holistic Smart Building Control*
[^evins-21]: Evins, 2021: *Building Energy Data for Machine Learning*
[^goyaletal-11]: Goyal et at., 2011: *Identification of multi-zone building thermal interaction model from data*
[^howardandgugger-20]: Howard and Gugger, 2021: *Deep Learning for Coders with fastai & PyTorch*
[^lari-21]: Lari, 2021: *Shaded Window Models Energy Plus*
[^machorrocanoetal-20]: Machorro-Cano et al., 2020: *HEMS-IoT: A Big Data and Machine-Learning Based Smart Home System for Energy Saving*
[^nrcan-19]: Natural Resources Canada, 2019: *Energy Use Data Handbook*
[^openstudio-21]: OpenStudio, 2021: *Open Studio SDK User Docs: About Measures*
[^unmethours-21]: UnmetHours, 2021: *How do you model operable windows?*
[^useia-20]: U.S. Energy Information Administration, 2020: *Energy Use Data Handbook*
[^wangandhong-20]: Wang and Hong, 2020: *Reinforcement learning for building controls: The opportunities and challenges*
[^zeilerandfergus-13]: Zeiler and Fergus, 2013: *Visualizing and Understanding Convolutional Networks*
[^zhangetal-21]: Zhang et at., 2021: *On the Joint Control of Multiple Building Systems with Reinforcement Learning*
[^zhouetal-17]: Zhou et at., 2017: *Quantitative Comparison of data-driven and physics-based models for commercial building HVAC systems*
