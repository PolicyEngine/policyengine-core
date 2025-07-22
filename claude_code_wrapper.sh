#!/bin/bash

# Define the log file
LOGFILE="claude_code_output.log"

# Run Claude Code and redirect both stdout and stderr to the log file
# while also displaying them in the terminal
claude "$@" > >(tee -a "$LOGFILE") 2> >(tee -a "$LOGFILE" >&2)
