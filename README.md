

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

## Feature and target selection 

The approach was guided by several questions: 

Is it possible to correlate the result of an F1 race with data regarded the strategies followed and the decisions made by teams in past races? Can we use race strategies and decisions taken by teams to gain valuable insights into which strategies might work best in the future? The aim is to be able to train a supervised machine learning model that would predict the winner outputs based on the best strategies.  With this in mind, we could develop a model that would help us identify the most effective strategies and decisions to achieve the best results in upcoming F1 races.

Therefore, the target aims to represent the success vs no-success into the strategy followed during a race for a driver/constructor pair team. And the more direct outcome is the result at the race. for this minimum viable product, the ability into predicting good strategic decisions will be related to the ability into distinguishing the results at the podium (1st, 2nd, 3rd) position, considered success, from the other results: no-success. The model, then, will be set as a binary classification problem, where It will be emphasized into predicting the podium correctly.

Winning results are made of a combination of highly-performant drivers, car and constructor teams, and good strategical decisions. After performing the exploratory data analysis of the primary datadabse, the data scrapping and the data transformation, the features selected for the dataset were:

- pitstops: withouth any doubt, if we want to study the strategy outcome of F1 races, we have to considerate the `pitstops number` in a race for a car, and a feature representing the `time performance on pitstops`. The last was calculated following a rolling average for the last ten results per constructor.

- driver skill: 'driver’s skills to win races' (driver from now on) could be parameterized based on historical performance metrics.
But those performance metrics should be independent of the several changes happended since 1950 at the sport, to name a few: the car layout and technologie, the FIA rules, the circuit layouts, if we want to compare drivers at all times. Then, the simpler approach that is independent of historical changes is to describe the driver's skill as a vector, with every component representing the propotions a driver got to a specific position at F1 races compared to the total races considered.  Then, those vectors were used to cluster the drivers based on their skills to win, and the results of that ML unsupervised machine learning was used as a categorical feature for the ML supervised model dataset.

- circuit: circuits layout and technical features could definitively impact a race strategy, e.g: material used in the track, number and types of corners, number and longitude of flat-out sections, can influence the grip and the tires degradation. Therefore the tires type selected and the times in pitstops are highly influenced by the circuit. The circuit variables included in the dataset for the 33 circuits involved are `circuit type`, categorical variable that classify the circuits between “street circuits” and “race circuits” where the former are those circuits that used public roads, like the famous circuit in Monaco, and the latter those circuit specially designed for sport races; `lap lenght`,  `Full throttle percentage`, `Downforce level`, and `Relative longest flat out lenght`.

- race results: if the driver finished the race, this number corresponds to the total laps for that specific circuit. If the driver did not finished the race, the total laps corresponds to the laps the driver was able to do before the incident happens.

- Constructor: The engine and or cars technical parameters will be indirectly considered by the inclusion of the categorical variable “constructor”, as the category that describes different car + engine independent pairs. The categorical variable was pre-treated following a one-hot encoding approach. 

- tire types: Up to 2022 there were a total of 5 different tire components for dry track weather conditions, and 2 tire types with different components for wet/rainy conditions. Per a race, 3 out of the 5 available tire components are nominated to be chosen by constructors during the race, in addition to the two possible wet conditions tire type, always available for the teams in case of changes in weather conditions. The 5 type of tires are named C1, C2, C3, C4 and C5, and the main difference between them is hardness, with C1 being the hardest and C5 the softest. 

- strategy: Strategy is described for our purpose by the plan of pit-stops and tire selection. Both tire type and stint (defined as the total number of laps a tire is used before the next pit-stop) are part of the strategy, and the order in which those stints and tire selections occur must be understood by the model. For example, if stint and tire type are included as independent columns in the dataset, it would not be obvious to the model, that the number in column “stint1” would correspond to the tire use “soft” in a feature column “tire type”. In other words, the pair “stint-tire type” are not independent to each other. To present the features appropriately,  I chose a numerical approach: several features were designed and named as combination of stint and tire type, e.g.: one of the columns can be the: “stint 1-soft”, another column: “stint 1-medium”, other would be the “stint 2-hard”, etc. The data in that column is the number of laps that stint lasted or 0 if that combination was not used by the driver of that race.

# Clustering drivers based in their ability to win races

## dataset preparation and experiments description

For the dataset preparation, were considered 855 drivers results from the primary Ergast database, and kept those with a minimum of 30 races. The initial dataset, then, was composed of 13273 rows of 216 drivers describing the final position per race, per every race registered since 1955 up to 2021. The results from 2022 were skept form the dataset because those races will be used to test the supervised ML, then I need it to avoid any leak of future results at the trainig dataset. The columns of the dataset are the proportion per every driver to gain the 1st, 2nd, 3rd, `post-podium`(counting the races a driver finished  4th and beyond up to the second-to-last), `second-to-last`(times a driver finished races before the ’last’ positions), `last` (total times a driver fot at the three last positions), considering all races results. The last ranges were designed to reduce the dimensionality of the dataset considering every race's outcome is composed of about 20 different position results, and the 1stm 2nd, 3rd position were considered individually because those columns are informative representing different probabilities per driver based in their ability to be the winners. 

After standarizing the dataset by removing the mean and scaling to unit variance, I trained different k-mean models considering different number of clusters to find the optimal, based on the average Silhouette per model, the sample distribution of the silhouette score per cluster, and the Calinski and Harabasz score, also known as the Variance Ratio Criterion.

- Silhouette coefficients near +1 indicate that the sample is far away from the neighboring clusters, therefore, scores near to 1 indicates well defined and separated clusters;

- A value of 0 indicates that the sample is on or very close to the decision boundary between two adjacent groups;

- Negative values indicate that those samples might have been assigned to the wrong cluster, or that the number of
clusters is incorrect.

The Calinski and Harabasz score is defined as the ratio of the sum of between-cluster dispersion and within-cluster dispersion. A higher Calinski-Harabasz score relates to a model with better-defined clusters.

Once the choice of the most appropriate number of clusters was done, the K-means model was trained again and the labels for the samples were obtained together with the histogram of the sample distribution of every label silhouette scoring to evaluate in more detail every cluster group. Finally, and in addition to the analysis based in scores, a comparison with an external source was performed to further analyze how meaningful were the groups obtained to cluster the drivers based in their skill.

## Results

The maximum average silhouette score was obtained for 2 clusters, but as we want to get a number of clusterings more meaningful than just splitting between “loosers” and “winners” this result was not considered. The second best-scored number of clusters were 5 to both scores, and it was the number of clusters selected to train the k-means and label the data.

The 5 (five) clusters obtained are named by chosing a representative famous driver, and after further analysis of the composition of those clusters, could be described as such:

- `Juan Manuel Fangio`: This cluster has the least number of pilots, and represents the elite F1 driver performers and includes, besides the one chosen to name the cluster, the historical drivers Ayrton Senna, Michael Schumacher, Alberto Ascari, Jackie Stewart, among others.

- `Fernando Alonso`: This cluster is composed of 50 drivers and is named “cluster 2” in the figures. It includes Carlos Reutemann, Valtteri Bottas, Kimi Raikkonen, Sebastian Vettel, Nino Farina, Max Verstappen, among others. These are the drivers that seem to be “The Champions,” those who have performed or are performing very well at F1 races and championships, but not (yet) as bright as those drivers in the Juan Manuel Fangio cluster.

- `Nicolas Hulkenberg`: Nicolas is the driver with the greatest number of appearances in Formula 1 without ever having been on a podium. This seems to be a good descriptor of the cluster, composed by drivers that had few or none have at all results at the podium, but more frequently at `post-podium` positions and contains 58 drivers from the training dataset, including Giancarlo Fisichella, Carlos Sainz, Sergio Perez, Felipe Massa, among others.

- `George Russell`: This cluster contains 62 drivers from the training dataset, and includes Roberto Moreno, Christian Fittipaldi, Gabriele Tarquini, Bruno Giacomelli. These drivers were just sporadically or never on the podium, and more freqently at `second-to-last` positions.

- Cluster David Brabham: This cluster contains 33 drivers from the training dataset and includes Tiago Monteiro, David Brabham, Max Chilton, Christijan Albers, and Pascal Wehrlein. These drivers did not stand out at all.

The three compounds chosen for a race could be C1, C2, C3, or maybe C3, C4, C5 or perhaps C2, C3, C4, but never will “jump” hardness as for example the combination C1, C2, C5 is not a valid combination of tires nominated for a race. For this reason, the current rules allows the relative classification of the tires used in a certain race as: “soft”, “medium” and “hard” being “soft” the softer tire compared to those three selected, and following the same criteria for the others. In this way, for a race where the components available are C1-C2-C3, the label “hard” will be for C1, the medium will be C2, and the “soft” alternative will be C3, meanwhile in a race where the three components nominated are the C3-C4-C5, C3 will become the labeled as “hard”, therefore the C4 component the “medium”, and C5 the “soft”.

During the data scrapping that was performed to get the historical selected tires on races, it was found several different naming for the tires, for example some years the terms used for the tern in the race could be “ultra-soft”, “super-soft” and “soft” and others named C1, C2, C3 among other combinations. Relative hardness of the available tires is what is considered impactful for the options available at a race that is part of the strategy design, therefore to align the historical data, code was developed to relabel the tires in a race: the term assigned to describe the softer tire, was relabeled as: “soft”, the second-one as “medium” and the last one, “hard”. E.g: some races were using the named “ultra-soft”, “super-soft”, “soft” for the tern used in the same race, being renamed this way: for “super-soft” was changed to “soft”, “ultra-soft” to “medium”, and “soft” to “hard”. After the data transformation, the historical dataset obtained from the scrapping process were composed of rows per lap and driver with the information of the tyre type in every race.

## Results compared to external rankings as ground truth

The nature of clustering analysis, based on learning with unlabeled data, determines as a challenge to evaluate if the group of entities are meaningful on real scenarios, and this is why in general, it is important to validate the clusters with an “expert” in the field. Researching in the internet, I have found [fivethirtyeight blog](https://fivethirtyeight.com/), dedicated to polling and statistics results from politics and sport. From an article named [Who’s The Best Formula One Driver Of All Time?](https://fivethirtyeight.com/features/formula-one-racing/), they evaluate the F1 driver performances at all times by following the modified ELO system to rank them.

The ELO rating system is a method for calculating the relative skill levels of players in zero-sum games such as chess. The ELO system was invented as an improved chess-rating system over the previously used Harkness system,but is also used as a rating system in association football, American football, baseball, basketball, pool, table tennis, and various board games and sports.

In the fivethirtyeight article, explains that all drivers are assigned ELO ratings going into each qualifying session and race, which represent their form — along with that of their engine manufacturer, mechanics, pit crew and so forth — at that particular moment. 

Then, they rank the best drivers based on their elo system, and show the best 30 drivers at all times. Given the published ranking is aim to showcase the best drivers at all times, the drivers selected for comparison were the `Juan Manuel Fangio` and `Fernando Alonso` clusters from the clustering process. The comparison was performed by merging both tables and checking whose of drivers are part of both rankings.

The results showed that 12 out of the 13 drivers in the Juan Manuel Fangio cluster were also included in the ELO ranking. The only driver identified by k-means in this cluster who was not part of the ELO ranking was Tony Brooks. This driver had the minimum Silhouette score (0.097) of this group, which means that he was identified as proximate to other cluster centers as well. Therefore, it is reasonable that this driver was outside of the cluster based on his silhouette score.

Since the ELO ranking contains 30 members, I compared the remaining drivers with those in the `Fernando Alonso` cluster. The result was that 16 drivers from the Fernando Alonso cluster are also in the ELO system ranking. Therefore, 28 out of the 30 total drivers included in the reference article are labeled as part of either the Juan Manuel Fangio or Fernando Alonso clusters, which were identified as the best drivers using k-means clustering. The remaining two drivers in the ELO system are in the third-best cluster, the Nicolas Hulkenberg one. 

We can conclude that the clustering process was able to split drivers into valuable groups based on their skill to win, being the two clusters with the best drivers, comparable to external rankings that uses additional data, and more complex- classical algorithm to order the drivers. 

## Results Dashboard

[Power BI TFM Results Dashboard](https://app.powerbi.com/view?r=eyJrIjoiYTI3ZTE3NjItYWVkOS00M2UzLTg3MWYtZGNlMTZmODdmODM2IiwidCI6ImU3ZjUzZjNmLTYzNmItNDNhZC04MDdlLTU3Yzk2NmZmN2RiOCIsImMiOjh9)

 
