require("ggplot2")
require("data.table")
require("mgcv")
require("splus2R")

# Read the FFT data
d.wbt.fft <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")
# Fit the GAM model
gam.wbt <- gam(power ~ s(freq, bs="cs"), data=d.wbt.fft)
# Get predicted values on testing data
test.wbt <- data.frame(freq = seq(0, 0.5, 0.01))
test.wbt$power <- predict(gam.wbt, test.wbt)

test.wbt$freq[peaks(test.wbt$power)]
test.wbt$power[peaks(test.wbt$power)]

# According to the inverse transform of DFT (https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
# x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \exp(2\pi i \frac{kn}{N})
