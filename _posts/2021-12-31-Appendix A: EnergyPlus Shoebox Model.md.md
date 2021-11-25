# Appendix A: EnergyPlus Shoebox Model
An appendix to *Automation and Machine Learning for Grid-Power Use Minimization in Sustainable Residential Architecture*

## Table of Contents

* [A.1 Introduction](#section_a_1)
* [A.2 EnergyPlus OpenStudioModel Model Definitions](#section_a_2)
  * [A.2.1 Weather](#section_a_2_1)
  * [A.2.2 Schedules](#section_a_2_2)
  * [A.2.3 Construction](#section_a_2_3)
  * [A.2.4 Loads](#section_a_2_4)
  * [A.2.5 Space Types](#section_a_2_5)
  * [A.2.6 Geometry](#section_a_2_6)
  * [A.2.7 Facility](#section_a_2_7)
  * [A.2.8 Spaces](#section_a_2_8)
  * [A.2.9 Thermal Zones](#section_a_2_9)
  * [A.2.10 Output Variables](#section_a_2_10)
  * [A.2.11 Simulation Settings](#section_a_2_11)
  * [A.2.12 Measures](#section_a_2_12)
* [A.3 Buoyancy Driven Cooling](#section_a_3)
* [References Revisited](#references-revisited)

Version/Revision History:

Version/Revision | Date Published | Details
-----|-----|----- 
V00, Rev.00 | 2021-11-25 | Initial Draft


## A.1 Introduction <a class="anchor" id="section_a_1"></a>

An EnergyPlus shoebox model has been created to generate a dataset demonstrating the dependencies between relevant system parameters over the course of a year. 

The shoebox model is the simplest representation of the situation under study—a low-energy-consumption residential home—that preserves all critical elements: solar radiation uptake, buoyancy-driven cooling, and baseboard heating.

Three pieces of software are used in conjunction to create this model:
* EnergyPlus – the nodal thermal simulation engine.
* Open Studio – a graphical user interface 
* SketchUp – a computer-aided design (CAD) program used to define and alter the model’s geometry. 

## A.2 EnergyPlus OpenStudioModel Model Definitions <a class="anchor" id="section_a_2"></a>

### A.2.1 Weather <a class="anchor" id="section_a_2_1"></a>

### A.2.2 Schedules <a class="anchor" id="section_a_2_2"></a>

### A.2.3 Construction <a class="anchor" id="section_a_2_3"></a>

### A.2.4 Loads <a class="anchor" id="section_a_2_4"></a>

### A.2.5 Space Types <a class="anchor" id="section_a_2_5"></a>

### A.2.6 Geometry <a class="anchor" id="section_a_2_6"></a>

### A.2.7 Facility <a class="anchor" id="section_a_2_7"></a>

### A.2.8 Spaces <a class="anchor" id="section_a_2_8"></a>

### A.2.9 Thermal Zones <a class="anchor" id="section_a_2_9"></a>

### A.2.10 Output Variables <a class="anchor" id="section_a_2_10"></a>

### A.2.11 Simulation Settings <a class="anchor" id="section_a_2_11"></a>

### A.2.12 Measures <a class="anchor" id="section_a_2_12"></a>

## A.3 Buoyancy-Driven Cooling <a class="anchor" id="section_a_3"></a>

An important part of the thermal model for this study is the role of openingwindows on the interior environment of the building. It is know that if the windows are opened when the air inside the house is hotter than the air outside, the difference in densities will cause the hot air to rise out of the house and pull cold air from the exterior environment into the house. This phenomenon is referred to as “buoyancy-driven cooling.” [^evins-21] 

No deffinive method esists by which to do this, and other researchers have used various approaches to include buoyancy-driven cooling in their thermal models. A few of these methods are discussed on unmethours.com, a well-known forum for EnergyPlus. Two particularly relevant approaches to controlling “operable windows” are the following

* “Add Wind and Stack Open Area” Method: https://unmethours.com/question/20051/how-can-i-model-operable-windows-in-open-studio-111-and-its-schedule/
* "AirflowNetwork:MultiZone:Component:DetailedOpening:" method: https://unmethours.com/question/13/how-do-you-model-operable-windows/

## References Revisited <a class="anchor" id="references-revisited"></a>
[^evins-21]: Evins, 2021: *Building Energy Data for Machine Learning*


