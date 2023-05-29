

# F1 data-driven strategy designer

The FIA Formula One World Championship has been one of the most famous and challenging forms of competition worldwide since its inaugural season in 1950. One of the most popular topics in debates around a race is what strategy of the constructor’s team is a better choice for a race. Even if constructors invest a fortune to get the best cars, engineers, and drivers, a wrong strategy can make them waste the opportunity of being at the race’s podium. This challenge aims to elaborate on the study of F1 data to provide a strategy advisor based on machine learning that could be useful as a tool for desition making to the F1 team and constructors.

## Introduction

What is the best tire for the circuit, how many changes of tires would they do, and when is the best moment in a race for the car to go to pit stops? Are some inquiries that the constructor’s team demands to answer during a race, and the decision-making made along these questions can set the final story of success or failure for them in that particular race.
Guided by this, the content of this essay was inspired by using the historical F1 race data provided by the Ergast Database and additional external data to develop a minimum viable product that helps to decide the best tire selection strategy and pitstops for a race in a particular circuit. The hypothetical clients could be the prominent investors of the competition, for example, the constructors and their sponsorships.

## State of the art

F1 cars are some of the fastest in the world, making every race exciting and fun to watch. To ensure that drivers reach the highest possible performance, teams of engineers and technicians work behind the scenes to develop data-driven strategies for each race. Teams analyze telemetry data and study the current track conditions to build the best setup for the car and driver.

The primary dataset used, does not contain telemetry data, being these telemetry data as well as the technical car details proprietary from the constructors. For this reason, one of the main challenges would be to gain valued insights from relative simpler data, specially historical results in previous races, and additional data that I scrapped from public F1 blogs.

## Data Wrangling

The primary data source was obtained from the Ergast web page [Ergast web page](http://ergast.com/mrd/) and open source SQL database which contains the historical data since the first F1 championship held on 1955 up to last championship on 2022.

The additional data for the tyre strategies per race and pilot was mainly obtain from the blog [Races.net](https://www.racefans.net/)
As the webpage does not provide an API for data consultation, the scrapping was performed using a classical approach, based on Python methods with the help of BeautifulSoup and Requests libraries.

## Feature selection and data preparation

The approach was guided by several questions: 

Is it possible to correlate the result of an F1 race with data regarded the strategies followed and the decisions made by teams in past races? Can we use race strategies and decisions taken by teams to gain valuable insights into which strategies might work best in the future? The aim is to be able to train a supervised machine learning model that would predict the winner outputs based on the best strategies.  With this in mind, we could develop a model that would help us identify the most effective strategies and decisions to achieve the best results in upcoming F1 races.

After performing the exploratory data analysis of the primary datadabse, the data scrapping and the data transformation, the features selected for the dataset were:

- pitstops: withouth any doubt, if we want to study the strategy outcome of F1 races, we have to considerate the `pitstops number` in a race for a car, and a feature representing the `time performance on pitstops`. The last was calculated following a rolling average for the last ten results per constructor.

- driver skill: 'driver’s skills to win races' (driver from now on) could be parameterized based on historical performance metrics.
But those performance metrics should be independent of the several changes happended since 1950 at the sport, to name a few: the car layout and technologie, the FIA rules, the circuit layouts, if we want to compare drivers at all times. Then, the simpler approach that is independent of historical changes is to describe the driver's skill as a vector, with every component representing the propotions a driver got to a specific position at F1 races compared to the total races considered.  Then, those vectors were used to cluster the drivers based on their skills to win, and the results of that ML unsupervised machine learning was used as a categorical feature for the ML supervised model dataset.

- circuit: circuits layout and technical features could definitively impact a race strategy, e.g: material used in the track, number and types of corners, number and longitude of flat-out sections, can influence the grip and the tires degradation. Therefore the tires type selected and the times in pitstops are highly influenced by the circuit. The circuit variables included in the dataset for the 33 circuits involved are `circuit type`, categorical variable that classify the circuits between “street circuits” and “race circuits” where the former are those circuits that used public roads, like the famous circuit in Monaco, and the latter those circuit specially designed for sport races; `lap lenght`,  `Full throttle percentage`, `Downforce level`, and `Relative longest flat out lenght`.

- race results: if the driver finished the race, this number corresponds to the total laps for that specific circuit. If the driver did not finished the race, the total laps corresponds to the laps the driver was able to do before the incident happens.

- Constructor: The engine and or cars technical parameters will be indirectly considered by the inclusion of the categorical variable “constructor”, as the category that describes different car + engine independent pairs. The categorical variable was pre-treated following a one-hot encoding approach. 

- tire types: Up to 2022 there were a total of 5 different tire components for dry track weather conditions, and 2 tire types with different components for wet/rainy conditions. Per a race, 3 out of the 5 available tire components are nominated to be chosen by constructors during the race, in addition to the two possible wet conditions tire type, always available for the teams in case of changes in weather conditions. The 5 type of tires are named C1, C2, C3, C4 and C5, and the main difference between them is hardness, with C1 being the hardest and C5 the softest. 

The three compounds chosen for a race could be C1, C2, C3, or maybe C3, C4, C5 or perhaps C2, C3, C4, but never will “jump” hardness as for example the combination C1, C2, C5 is not a valid combination of tires nominated for a race. For this reason, the current rules allows the relative classification of the tires used in a certain race as: “soft”, “medium” and “hard” being “soft” the softer tire compared to those three selected, and following the same criteria for the others. In this way, for a race where the components available are C1-C2-C3, the label “hard” will be for C1, the medium will be C2, and the “soft” alternative will be C3, meanwhile in a race where the three components nominated are the C3-C4-C5, C3 will become the labeled as “hard”, therefore the C4 component the “medium”, and C5 the “soft”.

During the data scrapping that was performed to get the historical selected tires on races, it was found several different naming for the tires, for example some years the terms used for the tern in the race could be “ultra-soft”, “super-soft” and “soft” and others named C1, C2, C3 among other combinations. Relative hardness of the available tires is what is considered impactful for the options available at a race that is part of the strategy design, therefore to align the historical data, code was developed to relabel the tires in a race: the term assigned to describe the softer tire, was relabeled as: “soft”, the second-one as “medium” and the last one, “hard”. E.g: some races were using the named “ultra-soft”, “super-soft”, “soft” for the tern used in the same race, being renamed this way: for “super-soft” was changed to “soft”, “ultra-soft” to “medium”, and “soft” to “hard”. After the data transformation, the historical dataset obtained from the scrapping process were composed of rows per lap and driver with the information of the tyre type in every race.

## Repository organization

 
