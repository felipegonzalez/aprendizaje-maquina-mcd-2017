#!/bin/sh
Rscript -e "devtools::install_version('glmnet', version = '2.0-10', repos = 'http://cran.us.r-project.org')"
Rscript -e "keras::install_keras()"
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"
