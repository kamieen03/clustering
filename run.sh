#!/usr/bin/env zsh
cd algs && \
cargo build --release && \
cd .. && \
./algs/target/release/algs && \
./process_output_data.py
