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

# Get GAM coefficients in data.table
get_gam_coefs <- function(file_name, norm=FALSE) {
  d <- fread(file_name)
  if (norm) {d$power <- d$power / max(d$power)} # normalize power
  gam <- gam(power ~ s(freq, bs="cs"), data=d)
  d.gam <- data.table(coef = gam$coefficients[2:length(gam$coefficients)],
                      order = 1:9)
  d.gam
}

# News
d.gam_gs_news0 <- get_gam_coefs("../data/gs_james/gs_news/webtext.train.model=.news_0.fft.csv")
d.gam_gs_news0$genre <- "news"
d.gam_gs_news0$model <- "gs"

# Story
# Wiki

# Try old gpt2 generated data for prelimanary comparison on GAM coef plot
d.gam_gpt2sm <- get_gam_coefs("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")
d.gam_gpt2sm$genre <- "news"
d.gam_gpt2sm$model <- "gpt2-small"
d.gam_gpt2md <- get_gam_coefs("../data/data_gpt2_old/medium-345M.test.model=gpt2-medium.fft.csv")
d.gam_gpt2md$genre <- "news"
d.gam_gpt2md$model <- "gpt2-medium"
d.gam_gpt2lg <- get_gam_coefs("../data/data_gpt2_old/large-762M.test.model=gpt2-large.fft.csv")
d.gam_gpt2lg$genre <- "news"
d.gam_gpt2lg$model <- "gpt2-large"
d.gam_gpt2xl <- get_gam_coefs("../data/data_gpt2_old/xl-1542M.test.model=gpt2-xl.fft.csv")
d.gam_gpt2xl$genre <- "news"
d.gam_gpt2xl$model <- "gpt2-xl"


# Bloomz-560m data
d.gam_bloomz560m_news <- get_gam_coefs("../data/data_bloomz_560m/webtext.train.model=.bloom_560m.news.fft.csv")
d.gam_bloomz560m_news$genre <- "news"
d.gam_bloomz560m_news$model <- "bloomz-560m"


# Combined
d.gam_comp <- rbindlist(list(d.gam_gs_news0, d.gam_bloomz560m_news,
                              d.gam_gpt2sm, d.gam_gpt2md, d.gam_gpt2lg, d.gam_gpt2xl))
p <- ggplot(d.gam_comp, aes(order, coef)) +
  geom_line(aes(color=model, linetype=model)) +
  geom_point(aes(color=model, shape=model)) +
  scale_x_discrete(limits = as.factor(1:9)) +
  theme_bw()
ggsave("bloomz-560m_gs-news_gpt2old.gam.coef.pdf", plot=p)