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
V01, Rev.00 | 2021-12-23 | Midterm Submission


## A.1 Introduction <a class="anchor" id="section_a_1"></a>

An EnergyPlus shoebox model has been created to generate a dataset demonstrating the dependencies between relevant system parameters over a year. The final model used to obtain the training and validation datasets for this study is a version of a ShoeBox model provided by Evins [^evins-21]  and Lari [^lari-21], altered to meet the current needs. It is the simplest representation of the situation under study—a low-energy-consumption residential home—that preserves all critical elements: solar radiation uptake, buoyancy-driven cooling, and baseboard heating.

Three pieces of software are used in conjunction to create this model:
* EnergyPlus – the nodal thermal simulation engine.
* Open Studio – a graphical user interface 
* SketchUp – a computer-aided design (CAD) program used to define and alter the model’s geometry. 

## A.2 EnergyPlus OpenStudioModel Model Definitions <a class="anchor" id="section_a_2"></a>

The subsections bellow are organized to match the organization of EnergyPlus functionality in the Open Studio graphical user interface. For each subsection there is a brief description as well as some screen shots explaining how that functionality was used. 

### A.2.1 Weather <a class="anchor" id="section_a_2_1"></a>

Historical weather data for Victoria, British Columbia is imported as “.epw” file, which defines the environmental conditions outside of the building for the study period. [^lari-21]

![](/images/20211200_MECH498_EnergyPlusModel_Weather_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Weather")

### A.2.2 Schedules <a class="anchor" id="section_a_2_2"></a>

Schedules are defined for occupancy, activity, lighting, electrical equipment, and infiltration. Each of these was imported as a boiler-plate schedule set for commercial office buildings modeled in EnergyPlus. 

![](/images/20211200_MECH498_EnergyPlusModel_Schedules_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Weather")

Cooling and Heating Setpoint schedules were also predefined in the model supplied by Evins [^evins-21] and Lari [^lari-21].

A schedule to control a Venetian blind covering either window was developed specifically for this study where blinds have an influence on radiative heat flux entering the building. Either Venitian blind can be either open or closed. Accordingly, an on/off schedule type is used to control them. An effort is made to schedule the blinds to be open during the winter daylight hours and closed during what will likely be the hottest daylight hours of the summer, but no ‘hard-and-fast’ rules are applied in defining this schedule. This schedule controls only the south-facing window which is exposed to the most sun. 

![](/images/20211200_MECH498_EnergyPlusModel_Schedules_VenetianBlinds_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Schedules")

### A.2.3 Construction <a class="anchor" id="section_a_2_3"></a>

Construction is relatively arbitrary for this study. It will impact absolute results but not relative ones between dependant variables. Knowing this, common constructions, predefined by Evins [^evins-21] and Lari [^lari-21], are left unaltered.

![](/images/20211200_MECH498_EnergyPlusModel_Construction_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Construction")

### A.2.4 Loads <a class="anchor" id="section_a_2_4"></a>

Loads may be defined for people, lights, internal mass, and various types of equipment. Again, for the sake of this study these parameters have absolute and not relative effects and are left unaltered from Evins’ [^evins-21] and Lari’s [^lari-21] original model.

![](/images/20211200_MECH498_EnergyPlusModel_Loads_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Loads")

### A.2.5 Space Types <a class="anchor" id="section_a_2_5"></a>

Space types can be associated with a default construction set, default schedule set, design specification for outdoor air, space infiltration flow rate, and space infiltration effective leakage areas. Evins’ [^evins-21] and Dr. Lari’s [^lari-21] original model is left unaltered, which defined only one office space type with base construction, office schedule set, and office outdoor air. 

![](/images/20211200_MECH498_EnergyPlusModel_Spaces_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Space Types")

### A.2.6 Geometry <a class="anchor" id="section_a_2_6"></a>

The geometry of a model is defined in the format of a “.idf” file. [^lari-21] Although the contents of these text files can be edited directly, it is often easiest to work on the geometry with the CAD software SketchUp.

The dimensions and orientation for the current shoebox model are comprised of two five-meter-long, by five-meter-wide, by three-meter-high (5m l × 5m w × 3m h) boxes stacked one on top of the other. Each has a four-meter-wide, by 1.5-meter-high (4m w × 1.5m h) window centered, one meter of the ground. The window for the lower box is in the south-facing wall, while the window for the upper box is in the north-facing wall. No floor/ceiling separates the lower box from the upper; however, they are defined as separate “thermal zones.”

![](/images/220211200_MECH498_EnergyPlusModel_Geometry_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Geometry")

### A.2.7 Facility <a class="anchor" id="section_a_2_7"></a>

The facility function of energy plus is not used for the purposes of this model.

![](/images/20211200_MECH498_EnergyPlusModel_Facility_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Facility")

### A.2.8 Spaces <a class="anchor" id="section_a_2_8"></a>

The spaces subsection of the OpenStudio EnergyPlus interface allows properties, loads, surfaces, subsurfaces, interior spaces, and shading to be assigned to different spaces. Two relevant specifications made for the two spaces in the current model are to the spaces themselves as well as the window subsurfaces.

In the current model, the first and second storeis are specified as separate spaces, each with its own associated thermal zone, but with the same space type, default construction type, and default schedule set.   

The windows are defined as “OperableWindows,” indicating that they are opened and closed throughout study. [^lari-21]

Likewise, the shading control type for the same windows is defined as “OnIfScheduleAllows,” indicating that the light transmission allowed by the associated Venetian blinds will be controlled by a schedule. [^lari-21]

Loads remain unaltered from the original shoebox model provided by Evins [^evins-21] and Lari [^lari-21], and the interior partition functionality is not used.

![](/images/20211200_MECH498_EnergyPlusModel_Spaces_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Spaces")

### A.2.9 Thermal Zones <a class="anchor" id="section_a_2_9"></a>

As stated in the ‘Geometry’ sub-section, two thermal zones are defined. This allows the temperature and temperature rate-of-change to be evaluated for both the hypothetical first floor, and the hypothetical second floor. 

![](/images/20211200_MECH498_EnergyPlusModel_ThermalZones_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Thermal Zones")

### A.2.10 Output Variables <a class="anchor" id="section_a_2_10"></a>

{% include alert.html text="Note: This part of the EnergyPlus Model is incomplete due to lack of clarity about which EnergyPlus output variables best meet the requirements of the model." %}

Using Big-Ladder Software’s [^bigladder-21] EnergyPlus Online Documentation, the outputs that corresponding to the relevant parameters being observed have been selected and tabulated below:

|          Parameter Type         |                             Parameter                            |                                               EnergyPlus Output                                               | EnergyPlus Output Desc. |
|---------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|-------------------------|
| Constraint                      | Temperature Setpoint                                             |                                                                                                               |                         |
| Control (Independent) Parameter | % closed (window 1)                                              |                                                                                                               |                         |
| Control (Independent) Parameter | % blinded (window 1)                                             |                                                                                                               |                         |
| Control (Independent) Parameter | % closed (window 1)                                              |                                                                                                               |                         |
| Control (Independent) Parameter | % blinded (window 1)                                             |                                                                                                               |                         |
| Dependant Parameter             | temperature (inside at the elevation of window 1)                |                                                                                                               |                         |
| Dependant Parameter             | temperature rate of change (inside at the elevation of window 1) |                                                                                                               |                         |
| Dependant Parameter             | temperature (inside at the elevation of window 1)                |                                                                                                               |                         |
| Dependant Parameter             | temperature rate of change (inside at the elevation of window 1) |                                                                                                               |                         |
| Invariraible Parameter          | Temperature (outside)                                            | Zone,Average,Site Outdoor Air Drybulb Temperature [C]                                                         |                         |
| Invariraible Parameter          | Temperature rate-of-change (outside)                             |                                                                                                               |                         |
| Invariraible Parameter          | Sun angle                                                        |                                                                                                               |                         |
| Invariraible Parameter          | Sun Intensity                                                    |                                                                                                               |                         |
| Parameter to be Minimized       | Artificial Heat Gain (HVAC)                                      | HVAC,Average,Zone Air System Sensible Heating Rate [W] HVAC,Average,Zone Air System Sensible Cooling Rate [W] |                         |

A few relevant notes, from Big-Ladder Software [^bigladder-21], about interpreting output variables:

> “Zone/HVAC - when the output is produced at the “Zone” timestep (ref: number of timesteps in each hour) or at the “HVAC” aka System timestep (which can vary for each hour).
Average/Sum - whether this is a averaged value over the reporting period (such as a temperature or rate) or whether this is a summed value over the reporting period. Reporting periods are specified in the Output:Variable or Output:Meter objects.
– The variable name one uses for reporting is displayed (e.g., Site Outdoor Drybulb Temperature) along with the units (e.g., [C]).”

![](/images/20211200_MECH498_EnergyPlusModel_OutputVariables_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Output Variables")

### A.2.11 Simulation Settings <a class="anchor" id="section_a_2_11"></a>

The relevant simulation settings are the date ranges and the number of time steps per hour. For this model, a full year with six time steps per hour is specified, resulting in 52560 data points in the final data set. 

![](/images/20211200_MECH498_EnergyPlusModel_SimulationSettings_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Simulation Settings")

### A.2.12 Measures <a class="anchor" id="section_a_2_12"></a>

{% include alert.html text="Note: This part of the EnergyPlus Model is unresolved. See subsection A.13 for more information." %}

Measures allow parametric studies to be completed in EnergyPlus by changing paramtedures during a simulation period to mimic real-world applications. [^bigladder-21] They are added to the current model for the purposes of understanding the parametric affects of varied amounts of window opening on the interior temperatures over the study period. This is discussed more in depth in the following subsection (A.13).

![](/images/20211200_MECH498_EnergyPlusModel_Measures_shradilkasseckert.jpg "EnergyPlus Shoebox Model: Measures")

## A.3 Buoyancy-Driven Cooling <a class="anchor" id="section_a_3"></a>

{% include alert.html text="Note: This part of the EnergyPlus Model is unresolved. See https://unmethours.com/question/66159/how-can-windows-be-controlled-so-that-they-allow-buoyancy-driven-cooling-in-a-multi-level-building/ for more information." %}

An important part of the thermal model for this study is the role of opening windows on the interior environment of the building. It is known that if the windows are opened when the air inside the house is hotter than the air outside, the difference in densities will cause the hot air to rise out of the house and pull cold air from the exterior environment into the house. This phenomenon is referred to as “buoyancy-driven cooling.” [^evins-21] 

No definitive method exists by which to do this, and other researchers have used various approaches to include buoyancy-driven cooling in their thermal models. 
The consensus among EneryPlus users on Unmet Hours, a well-known EnergyPlus forums, seems to be that “Measures” are the best way to model opening and closing windows. (UnmetHours, 2021) 

One particularly relevant approaches to controlling “operable windows” using measures is the following:
> “Add Wind and Stack Open Area” Method: https://unmethours.com/question/20051/how-can-i-model-operable-windows-in-open-studio-111-and-its-schedule/
However, multiple other approaches are discussed here as well: 
> “AirflowNetwork:MultiZone:Component:DetailedOpening” method: https://unmethours.com/question/13/how-do-you-model-operable-windows/

For the purposes of this study, the "Add Wind and Stack Open Area" method is used because it accurately models human bahviour and is well documented. Using this method, a maximum indoor temperature can be set above which the windows will be opened and buoyancy-driven cooling will occur. 

## References Revisited <a class="anchor" id="references-revisited"></a>
[^evins-21]: Evins, 2021: *Building Energy Data for Machine Learning*
[^lari-21]: Lari, 2021: *Shaded Window Models Energy Plus*
[^bigladder-21]: Big-Ladder Software, 2021: Energy Plus Web-Based Documentation
