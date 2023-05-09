require("data.table")
require("ggplot2")
require("stringr")

require("fpp2") # https://github.com/robjhyndman/fpp2-package
require("forecast")
require("urca") # https://xiaorui.site/Forecasting-and-Time-Series-Methods/Lecture/5_ADF.html
require("tseries") # This package provide adf.test()

# Example on a random data
x <- rnorm(1000)
adf.test(x)
# Dickey-Fuller = -9.9929, Lag order = 9, p-value = 0.01
# alternative hypothesis: stationary
# Conclusion: reject the null hypothesis, the series is stationary

# Find all .nll files in data folder
files_bloom_news <- list.files("../data/experiments_data/bloom/split_news/", pattern = "*.nll", full.names = TRUE)
files_bloom_story <- list.files("../data/experiments_data/bloom/split_story/", pattern = "*.nll", full.names = TRUE)
files_bloom_wiki <- list.files("../data/experiments_data/bloom/split_wiki/", pattern = "*.nll", full.names = TRUE)

files_opt_all <- list.files("../data/experiments_data/opt/", pattern = "*.nll", full.names = TRUE)
files_gpt2_all <- list.files("../data/experiments_data/gpt2/split_new/", pattern = "*.nll", full.names = TRUE)

files_webtext_old <- c("../data/data_gpt2_old/webtext.test.model=gpt2.nll")
files_gs_news <- list.files("../data/gs_james/gs_news/", pattern = "*.nll", full.names = TRUE)
files_gs_story <- list.files("../data/gs_james/gs_story/", pattern = "*.nll", full.names = TRUE)
files_gs_wiki <- list.files("../data/gs_james/gs_wiki/", pattern = "*.nll", full.names = TRUE)

# Read entropy data
read_one_nll <- function(file_name) {
  file_conn <- file(file_name, "r")
  lines <- readLines(file_conn)
  data <- data.table()
  for (i in 1:length(lines)) {
    entropy <- as.numeric(str_split(lines[i], " ")[[1]])
    seriesID <- rep(i, length(entropy))
    data <- rbindlist(list(data, data.table(seriesID = seriesID, entropy = entropy)))
  }
  data
}

read_multiple_nlls <- function (file_names) {
  data <- data.table()
  for (i in 1:length(file_names)) {
    tmp <- read_one_nll(file_names[i])
    if (i > 1) {
        tmp$seriesID <- tmp$seriesID + max(data$seriesID)
    }
    data <- rbindlist(list(data, tmp))
    print(paste0("Read ", i, " out of ", length(file_names)))
  }
  data
}

# Test stationary
test_one_dt <- function(dt) {
  dt.test <- data.table(series_id = numeric(),
                        series_len = numeric(),
                        adfpval = numeric()
  )
  # Suppress warning
  defaultW <- getOption("warn")
  options(warn = -1)
  unique_series_ids <- unique(dt$seriesID)
  for (i in 1:length(unique_series_ids)) {
    s_id <- unique_series_ids[i]
    entropy <- dt[seriesID == s_id]$entropy
    if (length(entropy) < 10) {next}
    adfpval <- adf.test(entropy)$p.value
    tmp <- data.table(series_id = s_id,
                      series_len = length(entropy),
                      adfpval = adfpval
    )
    dt.test <- rbindlist(list(dt.test, tmp))
    if (i %% 500 == 0) {
      write(paste0("Finished ", i, " out of ", length(unique_series_ids)), stdout())
    }
  }
  # Restore warning
  options(warn = defaultW)
  dt.test
}

# Test all bloom news data
dt.bloom_news <- read_multiple_nlls(files_bloom_news)
nrow(dt.bloom_news) #2993315
dt.bloom_news.test <- test_one_dt(dt.bloom_news)
nrow(dt.bloom_news.test[adfpval < 0.05]) / nrow(dt.bloom_news.test) # 0.7672 (76.7%)

# Test all bloom story data
dt.bloom_story <- read_multiple_nlls(files_bloom_story)
nrow(dt.bloom_story) # 3052753
dt.bloom_story.test <- test_one_dt(dt.bloom_story)
nrow(dt.bloom_story.test[adfpval < 0.05]) / nrow(dt.bloom_story.test) # 0.6680402 (66.8%)

# Test all bloom wiki data
dt.bloom_wiki <- read_multiple_nlls(files_bloom_wiki)
nrow(dt.bloom_wiki) # 3988866
dt.bloom_wiki.test <- test_one_dt(dt.bloom_wiki)
nrow(dt.bloom_wiki.test[adfpval < 0.05]) / nrow(dt.bloom_wiki.test) # 0.788 (78.8%)

# bloom total
(0.7672 * 2993315 + 0.6680402 * 3052753 + 0.788 * 3988866) / (2993315 + 3052753 + 3988866) # 0.7453023 (74.5%)

# Test all opt data
dt.opt <- read_multiple_nlls(files_opt_all)
nrow(dt.opt) # 27766156
dt.opt.test <- test_one_dt(dt.opt)
nrow(dt.opt.test[adfpval < 0.05]) / nrow(dt.opt.test) # 0.921 (92.1%)

# Test all gpt2 data
dt.gpt2 <- read_multiple_nlls(files_gpt2_all)
nrow(dt.gpt2) # 20680913
dt.gpt2.test <- test_one_dt(dt.gpt2)
nrow(dt.gpt2.test[adfpval < 0.05]) / nrow(dt.gpt2.test) # 0.9737667 (97.4%)

# Test old webtext data
dt.webtext_old <- read_multiple_nlls(files_webtext_old)
nrow(dt.webtext_old) # 2895278
dt.webtext_old.test <- test_one_dt(dt.webtext_old)
nrow(dt.webtext_old.test[adfpval < 0.05]) / nrow(dt.webtext_old.test) # 0.9634 (96.3%)

# Test gold standard (gs) news data
dt.gs_news <- read_multiple_nlls(files_gs_news)
nrow(dt.gs_news) # 8099091
dt.gs_news.test <- test_one_dt(dt.gs_news)
nrow(dt.gs_news.test[adfpval < 0.05]) / nrow(dt.gs_news.test) # 0.9605625 (96.1%)

# Test gs story data
dt.gs_story <- read_multiple_nlls(files_gs_story)
nrow(dt.gs_story) # 12791412
dt.gs_story.test <- test_one_dt(dt.gs_story)
nrow(dt.gs_story.test[adfpval < 0.05]) / nrow(dt.gs_story.test) # 0.9986 (99.9%)

# Test gs wiki data
dt.gs_wiki <- read_multiple_nlls(files_gs_wiki)
nrow(dt.gs_wiki) # 12469056
dt.gs_wiki.test <- test_one_dt(dt.gs_wiki)
nrow(dt.gs_wiki.test[adfpval < 0.05]) / nrow(dt.gs_wiki.test) # 0.97132 (97.1%)

# gs total
(0.9605625 * 8099091 + 0.9986 * 12791412 + 0.97132 * 12469056) / (8099091 + 12791412 + 12469056) # 0.9791685 (97.9%)






#### Old code
# dt.test <- data.table(series_id = numeric(),
#                       series_len = numeric(),
#                       boxpal = numeric(),
#                       adfpval = numeric(),
#                       kpsspval = numeric(),
#                       pppval = numeric())
# # Suppress warning
# defaultW <- getOption("warn")
# options(warn = -1)
# for (s_id in unique(dt$series_id)) {
#   entropy <- dt[series_id == s_id]$entropy
#   boxpval <- Box.test(entropy)$p.value
#   adfpval <- adf.test(entropy)$p.value
#   kpsspval <- kpss.test(entropy)$p.value
#   pppval <- pp.test(entropy)$p.value
#   tmp <- data.table(series_id = s_id,
#                     series_len = length(entropy),
#                     boxpal = boxpval,
#                     adfpval = adfpval,
#                     kpsspval = kpsspval,
#                     pppval = pppval)
#   dt.test <- rbindlist(list(dt.test, tmp))
# }
# # Restore warning
# options(warn = defaultW)
