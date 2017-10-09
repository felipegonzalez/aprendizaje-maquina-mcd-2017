#!/bin/sh
Rscript -e "keras::install_keras()"
Rscript -e "bookdown::render_book('index.Rmd', 'bookdown::gitbook')"
