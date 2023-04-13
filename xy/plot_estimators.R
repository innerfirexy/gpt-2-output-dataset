require("ggplot2")
require("data.table")
require("mgcv")


### Code from spectrum_plot.R
# Use some old data
old_data_dir <- "../data/data_gpt2_old/"

read_all_gpt2_files <- function(files) {
  models <- c("gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl")
  dt <- data.table()
  for (i in 1:length(files)) {
    d <- fread(files[i])
    d$model <- models[i]
    dt <- rbindlist(list(dt, d))
  }
  dt
}

webtext_names <- c("webtext.test.model=gpt2",
                 "webtext.test.model=gpt2-medium",
                 "webtext.test.model=gpt2-large",
                 "webtext.test.model=gpt2-xl")

fft_files <- paste(old_data_dir, webtext_names, sep = "", ".fft.csv")
d.fft <- read_all_gpt2_files(fft_files)

# Plot spectrum with each estimator model in a separate facet
p <- ggplot(d.fft, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model)) +
  facet_wrap(~model, ncol = 2) +
  labs(fill="Estimator model", linetype="Estimator model", colour="Estimator model") +
  theme_bw()
ggsave("webtext.4estimators.fft.4facets.pdf", plot=p)

# Obtain the smoothed curve using 'gam' method, y ~ s(x, bs = "cs")
gam_small <- gam(power ~ s(freq, bs = "cs"), data = d.fft[model=="gpt2-small"])
gam_small$coefficients
# (Intercept)   s(freq).1   s(freq).2   s(freq).3   s(freq).4   s(freq).5   s(freq).6   s(freq).7   s(freq).8   s(freq).9
# 10.47537  -112.59282   -47.32098   -78.75435   -62.55370   -71.04046   -66.98810   -66.03046   -75.08356   -30.50878
summary(gam_small)

testdata <- data.frame(freq = seq(0, 0.5, 0.01))
testdata$power <- predict(gam_small, testdata)
p <- ggplot(testdata, aes(freq, power)) +
  geom_line()
ggsave("webtext.gpt2-small.fft.gam.predictions.pdf", plot=p)