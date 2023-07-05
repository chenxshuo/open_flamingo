#!/bin/sh
session='monitor-new'

tmux new-session -d -s $session
# left
tmux splitw -h -p 15
tmux selectp -t 0
tmux send-keys "watch -n 1 'sinfo -o %35N%10P%10c%30G%10T%15C -e --sort=T'" C-m
tmux splitw -v -p 40
tmux send-keys "watch -n 1 'sinfo -R'" C-m
tmux splitw -v -p 40
tmux send-keys "watch -n 1 'squeue --format="%.8i%.28j%.8T%.15M%.15l%.20R" -u di93zun'" C-m

# right
tmux selectp -t 3
tmux send-keys "cd ~; source .bashrc; python watch_worker_lrz.py" C-m
tmux splitw -v -p 50
tmux send-keys "watch -n 1 'squeue --format=%13i%20j%15u%10T%10M%15l%20P%20R%Q --sort=Q -p mcml-dgx-a100-40x8'" C-m


tmux set -g mouse on

tmux attach-session -t $session
