#!/bin/bash

cd $(
    cd $(dirname $0)/../..
    pwd
)

MAX_T=8
MAX_Q=15

DEFAULT_M="./models/7B/ggml-model-q4_0.bin"
DEFAULT_T=4
DEFAULT_Q=4

GETOPTS_ARGS="t:q:m:bfh(help)"

function usage {
    options=${GETOPTS_ARGS//:/}
    echo "Usage: $(basename $0) -[$options]" 2>&1
    echo "  -t -threads   T number of threads [1, $MAX_T], default $DEFAULT_T"
    echo "  -q -questions Q number of questions [1, $MAX_Q], default $DEFAULT_Q"
    echo "  -m -model     M model file, default $DEFAULT_M"
    echo "  -b -bench       bench with number of questions from 1 to $MAX_Q, disabled by default"
    echo "  -f -filter      filter output by \"eval time =\", disabled by default"
    echo "  -h -help help   show this help"
    exit 1
}

T=$DEFAULT_T
Q=$DEFAULT_Q
M="$DEFAULT_M"
B=0
F=0

function parse_args {
    while getopts ${GETOPTS_ARGS} arg; do
        case "$arg" in
        t | threads) T="${OPTARG}" ;;
        q | questions) Q="${OPTARG}" ;;
        m | model) M="${OPTARG}" ;;
        b | bench) B=1 ;;
        f | filter) F=1 ;;
        h | help) usage ;;
        ?)
            echo "ERROR: invalid option: -${OPTARG}."
            echo
            usage
            ;;
        esac
    done

    if [[ "$T" -lt 1 ]] || [[ "$T" -gt $MAX_T ]]; then
        echo "ERROR: threads number out of range: $T"
        usage
    fi

    if [[ "$Q" -lt 1 ]] || [[ "$Q" -gt $MAX_Q ]]; then
        echo "ERROR: questions number out of range: $Q"
        usage
    fi

    if [ ! -e "$M" ]; then
        echo "ERROR: model file not exists: $M"
        usage
    fi
}

function prompt {
    p="" # prompt
    q=$1 # number of questions

    for ((i = 0; i < $q; i++)); do
        sum=$(($i + $i))
        p="${p}A:$i+$i=?\nB:$sum.\n"
    done

    model_name=$(basename $M)
    model_path="$(pwd)/$M"

    printf "=== model: %s, threads: %s, questions: %2d ===\n" $model_name $T $q

    args="-t $T -m $model_path \
		-c 512 -b 1024 --keep 256 --repeat_penalty 1.0 \
		--mlock --no-mmap \
		-n 8 -e -p $p"

    #echo "prompt" $p

    if [[ $F -eq 0 ]]; then
        ./main $args
    else
        ./main $args 2>&1 | grep "eval time ="
    fi

    exit_code=$?
    [[ exit_code -ne 0 ]] && {
        echo "error: main exited with $exit_code"
        #exit 1
    }
}

# __main__

parse_args "${@}"

if [[ $B -eq 0 ]]; then
    prompt $Q
else
    # bench
    for ((q = 1; q <= $MAX_Q; q++)); do
        prompt $q
    done
fi
