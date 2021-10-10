# Automation and Machine Learning for Grid Power-Use Minimization in Sustainable Residential Architecture



## Abstract

Text

## Table of Contense 

For ease in navigating the ramineder of this article, a table of contense:

{:toc}

## 1. Introduction

This is a (purely) theoretical efficacy study about achieving greater energy savings in sustainable residential architecture by applying automation and machine learning in a unique way. 

An industry is emerging in smart home design, and, although perhaps not conventional, one has existed in sustainable living solutions for a long time. While many engineers work on the problems surrounding energy storage, adressing energy consumpion can yield results that are just as impactful.


![Image of passive solar home](images/Passive Solar Home.jpg)

Consider a passive solar home (similar to the one shown in Figure ...). A homesteader designed it with an 80 by 7 foot south-facing wall of windows to allow solar radiation to pass into the home. Desirable amounts of heating and cooling are achieved by manually opening and closing blinds and windows. The process is repetitive and labor-intensive, which makes it an excellent candidate for automation. 

Deciding when to, and how much to, open the blinds and windows is not trivial. A great deal of experience, or knowledge about thermal behaviour is needed to make the right decision. This decision making process would be addressed well using one of two of today’s common engineering methods—(1) machine learning, or (2) a traditional analytical heat-transfer model.

Achieving higher efficiencies in homes that are already designed to save energy may be possible or even generally applicable. It is the objective of this study to form a conclusion about whether this can be accomplished using an automation and machine learning approach.

--

In this article,
- a theoretical thermal FEA model is created to simulate sustainable residential architecture,
- a control system is devised as a substitute for real-world automation of blinds, and windows, and 
- a machine learning algorithm is built and implemented to make experience-based control decisions about how to achieve the user's setpoint while mitigating grid power use.

Lastly, several data points are studied analytically, and results are compared to vailate the success of the machine learning algorithm.

“Sustainable Residential Architecture”
“Automation”
“Machine Learning”

## 2. Literature Review

Text

### 2.1 Previous Work on this Topic

Text

### 2.2 Why Automation is (or isn’t) Appropriate

Text

### 2.3 Why Machine Learning is (or isn’t) Appropriate

Howard and Gugger, in their 2020 publication Deep Learning for Coders with fastai & PyTorch, outline several inherent limitations of machine learning. Paraphrased, the relevant points are,


## 3. Method

Text

### 3.1 Defining (a) Thermal Model(s)

Historically, both experimental-apparatus’ and computer-driven simulations have been used for the validation of scientific theses, particularly in the field of engineering. In conversation, Evans, a building-energy researcher and instructor at the University of Victoria, explained that data has been a stumbling block in the building-energy research field. Often, data sets are available, but not enough information about where that came from is available to make it useful for all applications. Moreover, machine learning is inherently heavily data-dependent. In certain applications, like thermo- and fluid-dynamics more data can be obtained from computer-driven simulations than analogous experimental apparatus’ because of the absence of spatial sensor constraints. Hence, a computer-driven simulation model is used for this study. 

For the purposes of this study, the process of defining an appropriate first thermal model is one of simplifying a real-world scenario, while preserving all essential aspects. To do this, all critical parameters and dependencies are reviewed, and simplifying assumptions are applied to a real-world scenario:

Again, the objective of the final machine-learning model is to minimize the power introduced to a sustainable residential building through automated control of the windows and blinds while remaining within an error range around the user’s setpoint. 

All heat transfer into and out of the building has an influence on temperature regardless if by conduction, convection, or radiation. Convection is controlled by how open or closed the windows are. Radiation is controlled by how open or closed the blinds are. And the HVAC makes up for changes toward the user’s setpoint that cannot be achieved otherwise. 

Because of these dependencies, understanding what happens to all of these parameters when something inside the building is changed is critical:

- Constraints
  - temperature (user setpoint)
- Control (Independant) Parameters
  - % closed (window 1)
  - % blinded (window 1)
  - % closed (window 2)
  - % blinded (window 2) 
- Dependant Parameters
  - temperature (inside at the elevation of window 1)
  - temperature rate of change (inside at the elevation of window 1)
  - temperature (inside at the elevation of window 2)
  - temperature rate of change (inside at the elevation of window 2)
- Invariable Parameters
  - temperature (outside)
  - temperature rate of change (outside)
  - sun angle
  - sun intensity
- Parameter to be Minimizes
  - heat gain (due to the H-VAC system)
 
In the introduction of this article, a passively-solar-heated home was discussed. This residential building is quite complex, and considering all thermal behaviour would go far beyond requirements of a first model, which are simply to understand the influence of windows and blind on temperature within the building. For this reason, aspects such as thermal loads, an auxiliary solar water-heating system, internal geometry (eg. furniture), and humidity are ignored. Leftover is the fundamental geometry, an HVAC system, and set of windows and blinds. This can be simplified further to a “shoe-box” model consisting of just a two-story box with an HVAC system and windows on the first and second floors. Even with this, the impact of windows and blinds on internal temperature can be studied effectively.
 
Evans suggests using Energy Plus, a nodal building-energy simulator for constructing a computer-driven thermal.


### 3.2 Defining a Control System

Text

### 3.3 Building a Neural Network

Text

### 3.4 Training the Neural Network

According to Howard and Gugger, “the single most important and challenging issue when training for all… algorithms” is overfitting.

### 3.5 Validating the “Predictions”

Text

## 4. Results

Text

## 5. Discussion

Text

## 6. Summary and Conclusions

Text

## Acknowledgements

My honors thesis supervisors, 
- Dr. Rustom Bhiladvala, and 
- Dr. Yang Shi.

Other researchers who helped me significantly, 
- Dr. Ralph Evins,
- Dr. Khosro Lari, and
- Joel Grieco.



## References

Howard and Gugger, *Deep Learning for Coders with fastai & PyTorch*


[^1]: This is the footnote.
