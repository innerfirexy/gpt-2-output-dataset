require("ggplot2")
require("data.table")

# Show the estimator model does not affect the spectrum plot
data_dir <- "../data/data_gpt2_old/"

base_bames <- c("small-117M.test.model=gpt2",
                "small-117M.test.model=gpt2-medium",
                "small-117M.test.model=gpt2-large",
                "small-117M.test.model=gpt2-xl")


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


# Periodogram normalized
periodogram_normalized_files <- paste(
  data_dir,
  base_bames,
  sep = "",
  ".periodogram.normalized.csv")
d.periodogram.normalized <- read_all_gpt2_files(periodogram_normalized_files)
p <- ggplot(d.periodogram.normalized, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("small-117M.4estimators.periodogram.normalized.pdf", plot=p)

# Periodogram
periodogram_files <- paste(
  data_dir,
  base_bames,
  sep = "",
  ".periodogram.csv")
d.periodogram <- read_all_gpt2_files(periodogram_files)
p <- ggplot(d.periodogram, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("small-117M.4estimators.periodogram.pdf", plot=p)


# FFT normalized
fft_normalized_files <- paste(
  data_dir,
  base_bames,
  sep = "",
  ".fft.normalized.csv")
d.fft.normalized <- read_all_gpt2_files(fft_normalized_files)
p <- ggplot(d.fft.normalized, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("small-117M.4estimators.fft.normalized.pdf", plot=p)

# FFT
fft_files <- paste(
  data_dir,
  base_bames,
  sep = "",
  ".fft.csv")
d.fft <- read_all_gpt2_files(fft_files)
p <- ggplot(d.fft, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("small-117M.4estimators.fft.pdf", plot=p)
# log10(power^2)
p <- ggplot(d.fft, aes(freq, log10(power^2))) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("small-117M.4estimators.fft.log10.pdf", plot=p)


# Plot in polar coordinates
p <- ggplot(d.fft, aes(freq, log10(power^2))) +
  geom_smooth(aes(linetype = model, fill = model, colour = model)) +
    coord_polar()
ggsave("small-117M.4estimators.fft.polar.pdf", plot=p)

p <- ggplot(d.fft.normalized, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model)) +
    coord_polar()
ggsave("small-117M.4estimators.fft.normalized.polar.pdf", plot=p)

p <- ggplot(d.periodogram, aes(freq, log10(power))) +
  geom_smooth(aes(linetype = model, fill = model, colour = model)) +
    coord_polar()
ggsave("small-117M.4estimators.periodogram.polar.pdf", plot=p)


####
# Webtext results
####
base_names2 <- c("webtext.test.model=gpt2",
                 "webtext.test.model=gpt2-medium",
                 "webtext.test.model=gpt2-large",
                 "webtext.test.model=gpt2-xl")

# Periodogram
period_files <- paste(data_dir, base_names2, sep = "", ".periodogram.csv")
d2.period <- read_all_gpt2_files(period_files)
p <- ggplot(d2.period, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("webtext.4estimators.periodogram.pdf", plot=p)

# Periodogram, normalized
period_files <- paste(data_dir, base_names2, sep = "", ".periodogram.normalized.csv")
d2.period.normalized <- read_all_gpt2_files(period_files)
p <- ggplot(d2.period.normalized, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("webtext.4estimators.periodogram.normalized.pdf", plot=p)


# FFT
fft_files <- paste(data_dir, base_names2, sep = "", ".fft.csv")
d2.fft <- read_all_gpt2_files(fft_files)
p <- ggplot(d2.fft, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("webtext.4estimators.fft.pdf", plot=p)
# log10(power^2)
p <- ggplot(d2.fft, aes(freq, log10(power^2))) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("webtext.4estimators.fft.log10.pdf", plot=p)

# FFT, normalized
fft_files <- paste(data_dir, base_names2, sep = "", ".fft.normalized.csv")
d2.fft.normalized <- read_all_gpt2_files(fft_files)
p <- ggplot(d2.fft.normalized, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("webtext.4estimators.fft.normalized.pdf", plot=p)

# Plot in polar coordinates
p <- ggplot(d2.fft, aes(freq, log10(power^2))) +
  geom_smooth(aes(linetype = model, fill = model, colour = model)) +
    coord_polar()
ggsave("webtext.4estimators.fft.polar.pdf", plot=p)
