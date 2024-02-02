# apc-anonymizer

apc-anonymizer anonymizes automatic passenger counting (APC) results for public transit using differential privacy (DP).

apc-anonymizer takes a description of vehicle models as configuration and produces an anonymization profile for each of those vehicle models as output.
While creating the profiles takes significant computing effort and should be done in advance and only once per vehicle model, using the profiles is very fast and can be done in a streaming setting for [GTFS Realtime](https://gtfs.org/realtime/) or [SIRI](https://www.siri-cen.eu/) APIs.

## How to use

1. Create a YAML configuration file, e.g. `./configuration.yaml`, describing the vehicle models.
   Here is an example:

   ```yaml
   ---
   outputDirectory: "/output"
   vehicleModels:
     - outputFilename: "volvo-8908rle.csv"
       minimumCounts:
         EMPTY: 0
         MANY_SEATS_AVAILABLE: 6
         FEW_SEATS_AVAILABLE: 36
         STANDING_ROOM_ONLY: 46
         CRUSHED_STANDING_ROOM_ONLY: 84
         FULL: 110
       maximumCount: 126
     - outputFilename: "vdl-cites-lle-120-255.csv"
       minimumCounts:
         EMPTY: 0
         MANY_SEATS_AVAILABLE: 5
         FEW_SEATS_AVAILABLE: 28
         STANDING_ROOM_ONLY: 36
         CRUSHED_STANDING_ROOM_ONLY: 55
         FULL: 69
       maximumCount: 77
   ```

   You can also modify the inference settings in the configuration file but the defaults are sensible and should not be touched unless you know what you are doing.
   The configuration file must follow the [JSON schema](./src/apc_anonymizer/apc-anonymizer-schema.json) which describes all the configuration options.

1. Create the anonymization profiles in Docker, e.g.:

   FIXME: Change Docker image tag.

   ```sh
   mkdir --parents "./output" && \
   docker run \
     --env="APC_ANONYMIZER_CONFIG_PATH=/config/configuration.yaml" \
     --volume="./configuration.yaml:/config/configuration.yaml:ro" \
     --volume="./output:/output" \
     --rm \
     "ORGANIZATION-HERE/apc-anonymizer"
   ```

1. Use the CSV files describing the anonymization profiles in directory `./output` to anonymize APC counts.

## Anonymization profile

An anonymization profile is a matrix of probabilities given as a CSV file.
For example, here is an anonymization profile for the above vehicle model with `outputFilename` set to `vdl-cites-lle-120-255.csv`:

```csv
passenger_count,EMPTY,MANY_SEATS_AVAILABLE,FEW_SEATS_AVAILABLE,STANDING_ROOM_ONLY,CRUSHED_STANDING_ROOM_ONLY,FULL
0,0.9945442676544189,0.00545568997040391,0,0,0,0
1,0.9853664636611938,0.01463352888822556,0,0,0,0
2,0.961617648601532,0.03838236629962921,0,0,0,0
3,0.8987381458282471,0.1012618392705917,0,0,0,0
4,0.7283140420913696,0.2716859877109528,0,0,0,0
5,0.2707823216915131,0.7292176485061646,0,0,0,0
6,0.09974031150341034,0.9002596735954285,0,0,0,0
7,0.03793953731656075,0.9620604515075684,0,0,0,0
8,0.0141748059540987,0.98582524061203,0,0,0,0
9,0.005330529063940048,0.994669497013092,0,0,0,0
10,0.001984262140467763,0.9980157613754272,0,0,0,0
11,0.0007348262588493526,0.9992651343345642,0,0,0,0
12,0.0002678765740711242,0.9997320771217346,0,0,0,0
13,9.617868636269122e-05,0.9999037981033325,0,0,0,0
14,3.196327088517137e-05,0.9999680519104004,0,0,0,0
15,8.256777618953492e-06,0.9999917149543762,0,0,0,0
16,0,1,0,0,0,0
17,0,0.9999920129776001,7.945316610857844e-06,0,0,0
18,0,0.999968409538269,3.158608160447329e-05,0,0,0
19,0,0.999904990196228,9.500608575763181e-05,0,0,0
20,0,0.9997348189353943,0.0002651688409969211,0,0,0
21,0,0.9992768168449402,0.000723180128261447,0,0,0
22,0,0.9980317950248718,0.00196822127327323,0,0,0
23,0,0.9947455525398254,0.005254453048110008,0,0,0
24,0,0.9858505129814148,0.01414943300187588,0,0,0
25,0,0.9619612097740173,0.0380304828286171,8.27526946522994e-06,0,0
26,0,0.8987494707107544,0.1012182831764221,3.223448948119767e-05,0,0
27,0,0.7302741408348083,0.2696298360824585,9.601414058124647e-05,0,0
28,0,0.2709315121173859,0.7287983298301697,0.0002701870689634234,0,0
29,0,0.09980796277523041,0.8994600176811218,0.0007320215227082372,0,0
30,0,0.03731024637818336,0.960715115070343,0.001974658342078328,0,0
31,0,0.01394272316247225,0.9806990027427673,0.005358217284083366,0,0
32,0,0.005189845804125071,0.9804355502128601,0.01437460631132126,0,0
33,0,0.00194033735897392,0.9600164890289307,0.03804313018918037,0,0
34,0,0.0007244842709042132,0.8987023234367371,0.1005731523036957,0,0
35,0,0.0002656856959220022,0.726941704750061,0.2727925777435303,0,0
36,0,9.431142098037526e-05,0.2693384885787964,0.7305671572685242,0,0
37,0,3.160634514642879e-05,0.1014151126146317,0.8985532522201538,0,0
38,0,7.99901408754522e-06,0.03749722614884377,0.9624947905540466,0,0
39,0,0,0.01414564903825521,0.9858543276786804,0,0
40,0,0,0.005227940622717142,0.9947720170021057,0,0
41,0,0,0.001936464221216738,0.9980635046958923,0,0
42,0,0,0.0007161981775425375,0.9992837905883789,0,0
43,0,0,0.0002660927129909396,0.9997338652610779,0,0
44,0,0,9.537417645333335e-05,0.9998961687088013,8.402141247643158e-06,0
45,0,0,3.189696144545451e-05,0.9999359846115112,3.21137958962936e-05,0
46,0,0,8.290440746350214e-06,0.9998947381973267,9.69587781582959e-05,0
47,0,0,0,0.9997295737266541,0.0002704428334254771,0
48,0,0,0,0.9992697834968567,0.0007302440935745835,0
49,0,0,0,0.9980231523513794,0.001976834842935205,0
50,0,0,0,0.9947169423103333,0.005283080041408539,0
51,0,0,0,0.9859121441841125,0.01408783346414566,0
52,0,0,0,0.9620274901390076,0.03797248750925064,0
53,0,0,0,0.8992160558700562,0.1007839217782021,0
54,0,0,0,0.7315070033073425,0.2684930264949799,0
55,0,0,0,0.2719060182571411,0.7280939817428589,0
56,0,0,0,0.1002311855554581,0.8997688293457031,0
57,0,0,0,0.03762632980942726,0.962373673915863,0
58,0,0,0,0.0142809022217989,0.9857107996940613,8.242501280619763e-06
59,0,0,0,0.005332243163138628,0.9946359992027283,3.175332312821411e-05
60,0,0,0,0.001980263972654939,0.9979234933853149,9.628655243432149e-05
61,0,0,0,0.0007391276303678751,0.9989907145500183,0.0002701408811844885
62,0,0,0,0.000275332247838378,0.9989864230155945,0.0007382453186437488
63,0,0,0,9.849714115262032e-05,0.9979212284088135,0.001980270724743605
64,0,0,0,3.305606878711842e-05,0.994640588760376,0.005326365120708942
65,0,0,0,8.633205652586184e-06,0.9855883121490479,0.01440304517745972
66,0,0,0,0,0.9624178409576416,0.03758218884468079
67,0,0,0,0,0.8983173966407776,0.101682610809803
68,0,0,0,0,0.7310342192649841,0.2689657807350159
69,0,0,0,0,0.2719347178936005,0.7280652523040771
70,0,0,0,0,0.1005026921629906,0.8994973301887512
71,0,0,0,0,0.03738899156451225,0.9626109600067139
72,0,0,0,0,0.01396037731319666,0.9860396385192871
73,0,0,0,0,0.005289149936288595,0.9947108030319214
74,0,0,0,0,0.001943318406119943,0.9980567097663879
75,0,0,0,0,0.000730296946130693,0.9992696642875671
76,0,0,0,0,0.0002677132142707705,0.9997323155403137
77,0,0,0,0,9.838719415711239e-05,0.9999015927314758
```

The first column describes the amount of passengers onboard.
The rest of the columns represent the probabilities for the ordinal categories, e.g. GTFS Realtime OccupancyStatus values such as `EMPTY` or `STANDING_ROOM_ONLY`.

Each row describes the probability mass function for the category to publish given the amount of passengers onboard.
Using the example profile above, if there are 54 passengers, OccupancyStatus `STANDING_ROOM_ONLY` should be published with roughly 73 % probability and `CRUSHED_STANDING_ROOM_ONLY` should be published with roughly 27 % probability.

### How to use an anonymization profile

FIXME: check when to update if stop has no movement.

We have provided [example code](./src/apc_anonymizer/mechanisms/simple/sampling.py) in Python for sampling from an anonymization profile.
You can translate it to the language you need.

## How to develop

1. Install [Poetry](https://python-poetry.org/).
1. Run `poetry install`.

Run `./check.sh` to check the code quality.

FIXME: fix organization

Run `docker build --target=runtime --tag="ORGANIZATION-HERE/apc-anonymizer" .` to build the Docker image.

## Used by

[Waltti-APC](https://github.com/tvv-lippu-ja-maksujarjestelma-oy/waltti-apc), the Finnish national APC system, uses apc-anonymizer.

## Motivation and theory

Publishing precise APC results can easily reveal the movement of individuals with habitual travel patterns, thus violating their privacy.

A traditional approach to reduce the privacy leak has been to map the amount of passengers onboard onto ordinal categories so that the same amount of passengers onboard always results in the same category.
For example, if the passenger count ranges from 6 to 35 passengers in a certain vehicle model, we would always publish the occupancy status `MANY_SEATS_AVAILABLE`.

However, consider an individual that regularly boards a particular journey from a certain stop at a certain time.
Usually no one else boards and no one alights the vehicle at the same time.
Then if the precise count was 5 before the stop and 6 after the stop, the published change in the occupancy status may be connected to the individual's movement.

We should use differential privacy to fudge the transition from one category to the next.

### The privacy mechanism

apc-anonymizer is based on finding a probability table of releasing a vehicle state based on the number of passengers.
This probability table is found by optimizing the probability of releasing the correct category **while** satisfying differential privacy (DP) [1].
The loss is therefore comprised of two main parts: a log-likelihood for releasing the correct category $L_1$ (which we want to maximize), and a DP-cost function $L_2$ (which we want to minimize).
Additionally, the loss discourages the probability of releasing categories that are far from the truth, and we add another penalty $L_3$ for this.
The final loss will is given as $L = -c_1L_1 + c_2L_2 + c_3L_3$, where the $c_1, c_2$ and $c_3$ are positive coefficients that weight the importance of each loss.
For given $c_i$, the algorithm uses stochastic gradient descent to find the optimal probability table.
Additionally to find optimal coefficients $c_1, c_2, c_3$, we will use the [Optuna](https://github.com/optuna/optuna) package to optimize these hyperparameters.
Furthermore, we will have a hinge loss for the DP penalty which returns $\inf$ whenever the probability table breaks the desired DP guarantee (given as parameters epsilon and delta to the algorithm).

After the probability table is learned, it can be used to sample the categories based on the passenger count.
Note however, that the proposed algorithm provides desired DP guarantee only for a single release.
As the presense/absense of an individual might affect the stream of passenger counts at multiple times, the privacy protection should be ideally also provided for the entire stream and not only for a single release.

We draw the samples in a cryptographically secure manner.
Otherwise an attacker might be able to predict the produced randomness and break the privacy guarantee [2].

[1] Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006.  
[2] Simson L. Garfinkel and Philip Leclerc. Randomness concerns when deploying differential privacy. In Proceedings of the 19th Workshop on Privacy in the Electronic Society, WPES’20, page 73–86, New York, NY, USA, 2020.
