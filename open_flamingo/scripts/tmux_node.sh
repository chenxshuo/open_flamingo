#!/bin/sh
session='gpu-node'

tmux new-session -d -s $session
tmux set -g mouse on

tmux new-window -t $session:0 -n 'w0'
# left
tmux splitw -h -l 5
tmux selectp -t 0
tmux send-keys "source ~/.bashrc; cd $DATA" C-m

# right
tmux selectp -t 1
tmux send-keys "cd ~; source .bashrc; nvitop" C-m
tmux splitw -v -p 50
tmux send-keys "htop" C-m
tmux splitw -v -p 40
tmux send-keys "cd ~; source .bashrc; python set_gpus.py" C-m

tmux new-window -t $session:1 -n 'w1'
# left
tmux splitw -h -l 5
tmux selectp -t 0
tmux send-keys "source ~/.bashrc; cd $DATA" C-m

# right
tmux selectp -t 1
tmux send-keys "cd ~; source .bashrc; nvitop" C-m
tmux splitw -v -p 50
tmux send-keys "htop" C-m
tmux splitw -v -p 40
#tmux send-keys "cd ~; source .bashrc; python set_gpus.py" C-m


tmux new-window -t $session:2 -n 'w2'
# left
tmux splitw -h -l 5
tmux selectp -t 0
tmux send-keys "source ~/.bashrc; cd $DATA" C-m

# right
tmux selectp -t 1
tmux send-keys "cd ~; source .bashrc; nvitop" C-m
tmux splitw -v -p 50
tmux send-keys "htop" C-m
tmux splitw -v -p 40
#tmux send-keys "cd ~; source .bashrc; python set_gpus.py" C-m


tmux selectw -t $session:0
tmux attach-session -t $session
