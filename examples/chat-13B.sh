#!/bin/bash

cd "$(dirname "$0")/.." || exit

MODEL="${MODEL:-./models/13B/ggml-model-q4_0.bin}"
USER_NAME="${USER_NAME:-User}"
AI_NAME="${AI_NAME:-ChatLLaMa}"

# Adjust to the number of CPU cores you want to use.
N_THREAD="${N_THREAD:-8}"
# Number of tokens to predict (made it larger than default because we want a long interaction)
N_PREDICTS="${N_PREDICTS:-2048}"

# Note: you can also override the generation options by specifying them on the command line:
# For example, override the context size by doing: ./chatLLaMa --ctx_size 1024
GEN_OPTIONS="${GEN_OPTIONS:---ctx_size 2048 --temp 0.7 --top_k 40 --top_p 0.5 --repeat_last_n 256 --batch_size 1024 --repeat_penalty 1.17647}"

# shellcheck disable=SC2086 # Intended splitting of GEN_OPTIONS
./main $GEN_OPTIONS \
  --model "$MODEL" \
  --threads "$N_THREAD" \
  --n_predict "$N_PREDICTS" \
  --color --interactive \
  --reverse-prompt "${USER_NAME}:" \
  --prompt "
Text transcript of a never ending dialog, where ${USER_NAME} interacts with an AI assistant named ${AI_NAME}.
${AI_NAME} is helpful, kind, honest, friendly, good at writing and never fails to answer ${USER_NAME}’s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what ${USER_NAME} and ${AI_NAME} say aloud to each other.
The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
The transcript only includes text, it does not include markup like HTML and Markdown.

$USER_NAME: Hello, $AI_NAME!
$AI_NAME: Hello $USER_NAME! How may I help you today?
$USER_NAME: What time is it?
$AI_NAME: It is $(date +%H:%M).
$USER_NAME: What year is it?
$AI_NAME: We are in $(date +%Y).
$USER_NAME:" "$@"
