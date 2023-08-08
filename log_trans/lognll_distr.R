# Analyze the distributions of nll vs. log-transformed nll
require("ggplot2")
require("data.table")
require("stringr")
require("readr")

# The function for reading .nll and .nll_log data
read_nll_like_file <- function(file_path) {
    read_lines(file_path) %>%
        str_split(" ") %>%
        unlist() %>%
        as.numeric() %>%
        data.table(nll = .)
}

# Read raw .nll and .log_nll data
lognll.human <- read_nll_like_file("../data/data_gpt2_old/webtext.test.model=gpt2.nll_log")
lognll.gpt2sm <- read_nll_like_file("../data/data_gpt2_old/gpt2-sm.test.model=gpt2.nll_log")

nll.human <- read_nll_like_file("../data/data_gpt2_old/webtext.test.model=gpt2.nll")
nll.gpt2sm <- read_nll_like_file("../data/data_gpt2_old/small-117M.test.model=gpt2.nll")

lognll.human$source <- "human"
lognll.gpt2sm$source <- "gpt2sm"
lognll <- rbindlist(list(lognll.human, lognll.gpt2sm))
nll.human$source <- "human"
nll.gpt2sm$source <- "gpt2sm"
nll <- rbindlist(list(nll.human, nll.gpt2sm))

lognll.human$type <- "log-nll"
lognll.gpt2sm$type <- "log-nll"
nll.human$type <- "nll"
nll.gpt2sm$type <- "nll"

human <- rbindlist(list(lognll.human, nll.human))
gpt2sm <- rbindlist(list(lognll.gpt2sm, nll.gpt2sm))

# Plot the distributions
p <- ggplot(lognll, aes(nll)) + 
    geom_density(aes(fill=source, color=source), alpha=0.5) +
    ggtitle("Distribution of log-transformed nll") +
    theme_bw()
ggsave("lognll.density.(human+gpt2sm).pdf", plot=p)

p <- ggplot(nll, aes(nll)) + 
    geom_density(aes(fill=source, color=source), alpha=0.5) +
    ggtitle("Distribution of nll") +
    theme_bw()
ggsave("nll.density.(human+gpt2sm).pdf", plot=p)

# For same source plot nll and lognll together
p <- ggplot(human, aes(nll)) + 
    geom_density(aes(fill=type, color=type), alpha=0.5) +
    ggtitle("Distribution of human (nll & log-nll)") +
    theme_bw()
ggsave("human.density.(nll+lognll).pdf", plot=p)

p <- ggplot(gpt2sm, aes(nll)) + 
    geom_density(aes(fill=type, color=type), alpha=0.5) +
    ggtitle("Distribution of gpt2-sm (nll & log-nll)") +
    theme_bw()
ggsave("gpt2sm.density.(nll+lognll).pdf", plot=p)