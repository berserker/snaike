# snaike
A reinforcement learning-based simulation of the classic Snake game. This project explores how intelligent agents can learn to play Snake using modern RL algorithms in a simulated environment.

# setup
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# usage
python .\main.py --iterations=20 --timesteps=10000 --cores=8 --device=cuda --policy=MlpPolicy --grid_size=10 
