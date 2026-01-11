#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

# irace standard arguments
instance <- args[1]
seed <- as.integer(args[2])

# Remaining arguments are parameters
params <- list()

for (arg in args[-c(1, 2)]) {
  if (grepl("^--", arg)) {
    # Split --x4.968 or --x=4.968
    keyval <- sub("^--", "", arg)
    if (grepl("=", keyval)) {
      parts <- strsplit(keyval, "=")[[1]]
      params[[parts[1]]] <- parts[2]
    } else {
      # Split letters from numbers (e.g., x4.968)
      key <- sub("([a-zA-Z]+).*", "\\1", keyval)
      val <- sub("^[a-zA-Z]+", "", keyval)
      params[[key]] <- val
    }
  }
}

cmd <- sprintf(
  "python3 tuning.py --inst %s --seed %d --x %s --y %s --z %s",
  instance,
  seed,
  params$x,
  params$y,
  params$z
)

result <- system(cmd, intern = TRUE)
cat(result)
