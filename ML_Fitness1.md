### Apply Machine Learning Alogrithms for Establish Fitness Exercise Corectness

The goal of this project is to predict the manner in which they did the
exercise. To do this we are going to analze the data fro from this
source: <http://groupware.les.inf.puc-rio.br/har>.

Loading the required R Libraries.

    library(knitr)
    library(rmarkdown)
    library(markdown)
    library(caret)

    ## Loading required package: lattice
    ## Loading required package: ggplot2

    setwd("~/Documents/R/ML_fitness")
    library(ggplot2)
    library(rpart.plot)

    ## Loading required package: rpart

    library(corrplot)

Two datasets downloaed from the internet and then two local files
created.

    #files download
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
    destfile="training.csv")
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
    destfile="test.csv")

    #load data 
    training<-read.csv("training.csv")
    test<-read.csv("test.csv")
    head(training)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
    ## 1         no         11      1.41       8.07    -94.4                3
    ## 2         no         11      1.41       8.07    -94.4                3
    ## 3         no         11      1.42       8.07    -94.4                3
    ## 4         no         12      1.48       8.05    -94.4                3
    ## 5         no         12      1.48       8.07    -94.4                3
    ## 6         no         12      1.45       8.06    -94.4                3
    ##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
    ## 1                                                         
    ## 2                                                         
    ## 3                                                         
    ## 4                                                         
    ## 5                                                         
    ## 6                                                         
    ##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
    ## 1                                                                      NA
    ## 2                                                                      NA
    ## 3                                                                      NA
    ## 4                                                                      NA
    ## 5                                                                      NA
    ## 6                                                                      NA
    ##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
    ## 1             NA                         NA             NA             
    ## 2             NA                         NA             NA             
    ## 3             NA                         NA             NA             
    ## 4             NA                         NA             NA             
    ## 5             NA                         NA             NA             
    ## 6             NA                         NA             NA             
    ##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
    ## 1                  NA                   NA                   
    ## 2                  NA                   NA                   
    ## 3                  NA                   NA                   
    ## 4                  NA                   NA                   
    ## 5                  NA                   NA                   
    ## 6                  NA                   NA                   
    ##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
    ## 1                   NA            NA               NA            NA
    ## 2                   NA            NA               NA            NA
    ## 3                   NA            NA               NA            NA
    ## 4                   NA            NA               NA            NA
    ## 5                   NA            NA               NA            NA
    ## 6                   NA            NA               NA            NA
    ##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
    ## 1             NA                NA             NA           NA
    ## 2             NA                NA             NA           NA
    ## 3             NA                NA             NA           NA
    ## 4             NA                NA             NA           NA
    ## 5             NA                NA             NA           NA
    ## 6             NA                NA             NA           NA
    ##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
    ## 1              NA           NA         0.00         0.00        -0.02
    ## 2              NA           NA         0.02         0.00        -0.02
    ## 3              NA           NA         0.00         0.00        -0.02
    ## 4              NA           NA         0.02         0.00        -0.03
    ## 5              NA           NA         0.02         0.02        -0.02
    ## 6              NA           NA         0.02         0.00        -0.02
    ##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
    ## 1          -21            4           22            -3           599
    ## 2          -22            4           22            -7           608
    ## 3          -20            5           23            -2           600
    ## 4          -22            3           21            -6           604
    ## 5          -21            2           24            -6           600
    ## 6          -21            4           21             0           603
    ##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
    ## 1          -313     -128      22.5    -161              34            NA
    ## 2          -311     -128      22.5    -161              34            NA
    ## 3          -305     -128      22.5    -161              34            NA
    ## 4          -310     -128      22.1    -161              34            NA
    ## 5          -302     -128      22.1    -161              34            NA
    ## 6          -312     -128      22.0    -161              34            NA
    ##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
    ## 1           NA              NA           NA            NA               NA
    ## 2           NA              NA           NA            NA               NA
    ## 3           NA              NA           NA            NA               NA
    ## 4           NA              NA           NA            NA               NA
    ## 5           NA              NA           NA            NA               NA
    ## 6           NA              NA           NA            NA               NA
    ##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
    ## 1            NA          NA             NA          NA        0.00
    ## 2            NA          NA             NA          NA        0.02
    ## 3            NA          NA             NA          NA        0.02
    ## 4            NA          NA             NA          NA        0.02
    ## 5            NA          NA             NA          NA        0.00
    ## 6            NA          NA             NA          NA        0.02
    ##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
    ## 1        0.00       -0.02        -288         109        -123         -368
    ## 2       -0.02       -0.02        -290         110        -125         -369
    ## 3       -0.02       -0.02        -289         110        -126         -368
    ## 4       -0.03        0.02        -289         111        -123         -372
    ## 5       -0.03        0.00        -289         111        -123         -374
    ## 6       -0.03        0.00        -289         111        -122         -369
    ##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
    ## 1          337          516                                     
    ## 2          337          513                                     
    ## 3          344          513                                     
    ## 4          344          512                                     
    ## 5          337          506                                     
    ## 6          342          513                                     
    ##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
    ## 1                                                                       
    ## 2                                                                       
    ## 3                                                                       
    ## 4                                                                       
    ## 5                                                                       
    ## 6                                                                       
    ##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
    ## 1           NA            NA          NA           NA            NA
    ## 2           NA            NA          NA           NA            NA
    ## 3           NA            NA          NA           NA            NA
    ## 4           NA            NA          NA           NA            NA
    ## 5           NA            NA          NA           NA            NA
    ## 6           NA            NA          NA           NA            NA
    ##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
    ## 1          NA                 NA                  NA                NA
    ## 2          NA                 NA                  NA                NA
    ## 3          NA                 NA                  NA                NA
    ## 4          NA                 NA                  NA                NA
    ## 5          NA                 NA                  NA                NA
    ## 6          NA                 NA                  NA                NA
    ##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
    ## 1         13.05         -70.49       -84.87                       
    ## 2         13.13         -70.64       -84.71                       
    ## 3         12.85         -70.28       -85.14                       
    ## 4         13.43         -70.39       -84.87                       
    ## 5         13.38         -70.43       -84.85                       
    ## 6         13.38         -70.82       -84.47                       
    ##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
    ## 1                                                                     
    ## 2                                                                     
    ## 3                                                                     
    ## 4                                                                     
    ## 5                                                                     
    ## 6                                                                     
    ##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
    ## 1                                                              NA
    ## 2                                                              NA
    ## 3                                                              NA
    ## 4                                                              NA
    ## 5                                                              NA
    ## 6                                                              NA
    ##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
    ## 1                 NA                                 NA                 NA
    ## 2                 NA                                 NA                 NA
    ## 3                 NA                                 NA                 NA
    ## 4                 NA                                 NA                 NA
    ## 5                 NA                                 NA                 NA
    ## 6                 NA                                 NA                 NA
    ##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
    ## 1                                       NA                       NA
    ## 2                                       NA                       NA
    ## 3                                       NA                       NA
    ## 4                                       NA                       NA
    ## 5                                       NA                       NA
    ## 6                                       NA                       NA
    ##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
    ## 1                                          37                 NA
    ## 2                                          37                 NA
    ## 3                                          37                 NA
    ## 4                                          37                 NA
    ## 5                                          37                 NA
    ## 6                                          37                 NA
    ##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
    ## 1                NA                   NA                NA
    ## 2                NA                   NA                NA
    ## 3                NA                   NA                NA
    ## 4                NA                   NA                NA
    ## 5                NA                   NA                NA
    ## 6                NA                   NA                NA
    ##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
    ## 1                 NA                    NA                 NA
    ## 2                 NA                    NA                 NA
    ## 3                 NA                    NA                 NA
    ## 4                 NA                    NA                 NA
    ## 5                 NA                    NA                 NA
    ## 6                 NA                    NA                 NA
    ##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
    ## 1               NA                  NA               NA                0
    ## 2               NA                  NA               NA                0
    ## 3               NA                  NA               NA                0
    ## 4               NA                  NA               NA                0
    ## 5               NA                  NA               NA                0
    ## 6               NA                  NA               NA                0
    ##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
    ## 1            -0.02             0.00             -234               47
    ## 2            -0.02             0.00             -233               47
    ## 3            -0.02             0.00             -232               46
    ## 4            -0.02            -0.02             -232               48
    ## 5            -0.02             0.00             -233               48
    ## 6            -0.02             0.00             -234               48
    ##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
    ## 1             -271              -559               293               -65
    ## 2             -269              -555               296               -64
    ## 3             -270              -561               298               -63
    ## 4             -269              -552               303               -60
    ## 5             -270              -554               292               -68
    ## 6             -269              -558               294               -66
    ##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
    ## 1         28.4         -63.9        -153                      
    ## 2         28.3         -63.9        -153                      
    ## 3         28.3         -63.9        -152                      
    ## 4         28.1         -63.9        -152                      
    ## 5         28.0         -63.9        -152                      
    ## 6         27.9         -63.9        -152                      
    ##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
    ## 1                                                                  
    ## 2                                                                  
    ## 3                                                                  
    ## 4                                                                  
    ## 5                                                                  
    ## 6                                                                  
    ##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
    ## 1                                                           NA
    ## 2                                                           NA
    ## 3                                                           NA
    ## 4                                                           NA
    ## 5                                                           NA
    ## 6                                                           NA
    ##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
    ## 1                NA                               NA                NA
    ## 2                NA                               NA                NA
    ## 3                NA                               NA                NA
    ## 4                NA                               NA                NA
    ## 5                NA                               NA                NA
    ## 6                NA                               NA                NA
    ##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
    ## 1                                     NA                      NA
    ## 2                                     NA                      NA
    ## 3                                     NA                      NA
    ## 4                                     NA                      NA
    ## 5                                     NA                      NA
    ## 6                                     NA                      NA
    ##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
    ## 1                                        36                NA
    ## 2                                        36                NA
    ## 3                                        36                NA
    ## 4                                        36                NA
    ## 5                                        36                NA
    ## 6                                        36                NA
    ##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
    ## 1               NA                  NA               NA                NA
    ## 2               NA                  NA               NA                NA
    ## 3               NA                  NA               NA                NA
    ## 4               NA                  NA               NA                NA
    ## 5               NA                  NA               NA                NA
    ## 6               NA                  NA               NA                NA
    ##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
    ## 1                   NA                NA              NA
    ## 2                   NA                NA              NA
    ## 3                   NA                NA              NA
    ## 4                   NA                NA              NA
    ## 5                   NA                NA              NA
    ## 6                   NA                NA              NA
    ##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
    ## 1                 NA              NA            0.03            0.00
    ## 2                 NA              NA            0.02            0.00
    ## 3                 NA              NA            0.03           -0.02
    ## 4                 NA              NA            0.02           -0.02
    ## 5                 NA              NA            0.02            0.00
    ## 6                 NA              NA            0.02           -0.02
    ##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
    ## 1           -0.02             192             203            -215
    ## 2           -0.02             192             203            -216
    ## 3            0.00             196             204            -213
    ## 4            0.00             189             206            -214
    ## 5           -0.02             189             206            -214
    ## 6           -0.03             193             203            -215
    ##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
    ## 1              -17              654              476      A
    ## 2              -18              661              473      A
    ## 3              -18              658              469      A
    ## 4              -16              658              469      A
    ## 5              -17              655              473      A
    ## 6               -9              660              478      A

With data cleaning process we choose the colomuns that we are going to
use.

    include.col <- c("pitch_belt", "yaw_belt", "total_accel_belt", "gyros_belt_x", 
        "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", 
        "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", 
        "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", "gyros_arm_z", 
        "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", "magnet_arm_y", 
        "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", 
        "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", 
        "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", 
        "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
        "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", 
        "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", 
        "magnet_forearm_z")
    test1 <- test[, include.col]
    include.col<- c(include.col, "classe")
    training1 <- training[, include.col]
    dim(training)

    ## [1] 19622   160

    dim(training1)

    ## [1] 19622    52

Exploratory analysis
====================

Two figures produced for the data exploratory analysis. First a
histogram to count the frequency of the classe variable and second a
graph which shows that how different columns are correlated to each
other.

    qplot(training$classe, ylab="Frequency")

![plot of chunk
unnamed-chunk-4](./ML_Fitness1_files/figure-markdown_strict/unnamed-chunk-41.png)

    corr <- cor(training1[ ,-52])
    corrplot(corr, order="FPC", type="lower", method="color", tl.cex=0.8, tl.col="black")

![plot of chunk
unnamed-chunk-4](./ML_Fitness1_files/figure-markdown_strict/unnamed-chunk-42.png)

Prediction Model
================

We use the Randon Forest Algorithm for the following reasons. the
decisions tree algorithms adavatages are: Simple to understand and
interpret. People are able to understand decision tree models after a
brief explanation. -Requires little data preparation. -Able to handle
both numerical and categorical data. -Uses a white box model -Possible
to validate a model using statistical tests. That makes it possible to
account for the reliability of the model. -Robust. Performs well even if
its assumptions are somewhat violated by the true model from which the
data were generated.

    #Random Forest Algorithm
    library(randomForest)

    ## randomForest 4.6-10
    ## Type rfNews() to see new features/changes/bug fixes.

    model1<-randomForest(classe ~ ., data=training1, method="class")
    predict1<-predict(model1, training1, type="class")
    print(confusionMatrix(predict1, training1$classe))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 5580    0    0    0    0
    ##          B    0 3797    0    0    0
    ##          C    0    0 3422    0    0
    ##          D    0    0    0 3216    0
    ##          E    0    0    0    0 3607
    ## 
    ## Overall Statistics
    ##                                 
    ##                Accuracy : 1     
    ##                  95% CI : (1, 1)
    ##     No Information Rate : 0.284 
    ##     P-Value [Acc > NIR] : <2e-16
    ##                                 
    ##                   Kappa : 1     
    ##  Mcnemar's Test P-Value : NA    
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity             1.000    1.000    1.000    1.000    1.000
    ## Specificity             1.000    1.000    1.000    1.000    1.000
    ## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
    ## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
    ## Prevalence              0.284    0.194    0.174    0.164    0.184
    ## Detection Rate          0.284    0.194    0.174    0.164    0.184
    ## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
    ## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000

Testing
=======

    testing_res<-predict(model1,test1)
    testing_res

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Write Files
===========

    pml_write_files = function(x){
            n=length(x)
            for (i in 1:n){
              filename=paste0("problem_id_",i,".txt")
              write.table(x[i], file =filename, quote=FALSE, row.names=FALSE,col.names=FALSE)
            }
    }
    pml_write_files(testing_res)
