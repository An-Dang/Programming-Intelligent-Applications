call activate pia
start /b tensorboard --logdir=./tensorboard
timeout 10
start "" http://localhost:6006
pause
