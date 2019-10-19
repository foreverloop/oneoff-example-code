#question 1 population, sampling and sample sizes
# (a) population is the entire data set
# (b) a smaple that is too small does not capture enough variety in the data set
# a sample that is too big could be too big to manage, and computationally 
#expensive to obtain. sample size could be constrained by time, computation power
#and if the population itself is very small

#(c) random sample means to take data points from the data set at random
#without bias as to what is chosen

#question 2 lanarkshire milk experiment
#teachers were allowed to use their discretion in forming the groups
#this could easily introduce bias from the teacher, so this is one criticisim
#a second criticism is that the children might be wearing different clothes
#on different times, giving different weights. Also only weighing 2 times
#but possibly they could have weighed each month?

#question 3 destructive sampling
#capture rates decreased because the beetles which had already been captured 
#had been killed, reducing the population and therefore the chance of capturing
#this could be avoided by designing a non-lethal trap and releasing
#the beetles a good distance away from the traps

#question 4
#continuous - weight is a continuous variable
#discrete - grade obtained in first degree (pass, third, 2:2, 2:1, 1st)
#ordinal - infant/junior/adolescent/adult cats
#categorical (aka factor) - eye colour

#question 5
load("WorkshopData.Rdata")
#(a) what type of data is thamesFloods?
typeof(thamesFloods)
#(b) sample mean and variance
mean(thamesFloods)
var(thamesFloods)

#(c) index of dispersion (s^2 / x_bar)
var(thamesFloods) / mean(thamesFloods)

#(d) Are the data under- (D < 1) or over- (D > 1) dispersed relative
#to the Poisson case, in which D = 1?
#answer: the data are over, 1.714559 from line 39

#question 6
#(a) what type is oxfordRain variable?
typeof(oxfordRain)

#(b) Calculate the sample mean and sample variance of the rainfall totals. 
#What is the sample standard deviation, and how is it related to the variance?

mean(oxfordRain)
var(oxfordRain)
sd(oxfordRain) #standard deviation is the square root of the variance
sqrt(var(oxfordRain)) #proof

#(c) Is the sample median close to the sample mean? (yes, within 10 of it)
mean(oxfordRain)
median(oxfordRain)

#(d) By plotting a histogram of the rainfall totals, 
#explain whether the data show negative or positive skew.
hist(oxfordRain) #the data skews right, indicating a positive skewness

#(e) The R function density can be used to produce a smoothed 
#version of the histogram. Use the plots produced by this function 
#to estimate the mode of the rainfall totals.
help(density)
plot(density(oxfordRain)) #mode approx 650 according to this plot
