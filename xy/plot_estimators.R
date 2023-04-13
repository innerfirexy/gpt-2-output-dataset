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


####
# Golden standard data
data_dirs <- c("../data/gs_james/gs_news/",
               "../data/gs_james/gs_story/",
               "../data/gs_james/gs_wiki/")
genres <- c("news", "story", "wiki")
lengths <- c("0","1","2","3","4")

# News
d.gs_news0 <- fread("../data/gs_james/gs_news/webtext.train.model=.news_0.fft.csv")
gam_gs_news0 <- gam(power ~ s(freq, bs="cs"), data=d.gs_news0)
gam_gs_news0$coefficients
# (Intercept)   s(freq).1   s(freq).2   s(freq).3   s(freq).4   s(freq).5   s(freq).6   s(freq).7   s(freq).8   s(freq).9
# 10.02772   -84.57781   -31.45307   -56.91613   -44.49745   -52.38750   -49.73750   -50.95053    -57.17049   -25.40064
summary(gam_gs_news0)
d.gs_news0.gam <- data.table(coef = gam_gs_news0$coefficients[2:length(gam_gs_news0$coefficients)],
                             order = 1:9)
p <- ggplot(d.gs_news0.gam, aes(order, coef)) + geom_line()

d.gs_news1 <- fread("../data/gs_james/gs_news/webtext.train.model=.news_1.fft.csv")
gam_gs_news1 <- gam(power ~ s(freq, bs="cs"), data=d.gs_news1)
d.gs_news1.gam <- data.table(coef = gam_gs_news1$coefficients[2:length(gam_gs_news1$coefficients)],
                             order = 1:9)
p <- ggplot(d.gs_news1.gam, aes(order, coef)) + geom_line()

# Story
d.gs_story0 <- fread("../data/gs_james/gs_story/webtext.train.model=.story_0.fft.csv")
gam_gs_story0 <- gam(power ~ s(freq, bs="cs"), data=d.gs_story0)
d.gs_story0.gam <- data.table(coef = gam_gs_story0$coefficients[2:length(gam_gs_story0$coefficients)],
                             order = 1:9)
p <- ggplot(d.gs_story0.gam, aes(order, coef)) + geom_line()

# Wiki
d.gs_wiki0 <- fread("../data/gs_james/gs_wiki/webtext.train.model=.wiki_0.fft.csv")
gam_gs_wiki0 <- gam(power ~ s(freq, bs="cs"), data=d.gs_wiki0)
d.gs_wiki0.gam <- data.table(coef = gam_gs_wiki0$coefficients[2:length(gam_gs_wiki0$coefficients)],
                              order = 1:9)

# Combined
d.gs_news0.gam$genre <- "news"
d.gs_story0.gam$genre <- "story"
d.gs_wiki0.gam$genre <- "wiki"
d.gam <- rbindlist(list(d.gs_news0.gam, d.gs_story0.gam, d.gs_wiki0.gam))
p <- ggplot(d.gam, aes(order, coef)) +
  geom_line(aes(color=genre, linetype=genre))