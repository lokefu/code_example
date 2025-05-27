1. working directory: /home/jupyter/
simple run:
- cd `/home/jupyter/putr/PuTR`
- run `conda activate /home/jupyter/PuTR`
- run `python putr_single.py` or `python putr_batch.py`


2. structure
if no permission: run following and wait OR restart the server (change ip)

sudo chmod g+w /home/jupyter
sudo chmod g+s /home/jupyter
sudo apt update
sudo apt install acl -y
sudo setfacl -d -m g:jupyter:rwx /home/jupyter
sudo usermod -a -G jupyter "$USER"

- /home/jupyter/
  - putr/ #create folder
    - PuTR/ #git clone later
    - putr_single.py
    - putr_batch.py
    - dancetrack.pth
  - setup_putr_env.sh


3. run
- put `setup_putr_env.sh` under /home/jupyter
- run `chmod +x setup_putr_env.sh; source ./setup_putr_env.sh`
- put scripts & checkpoints
- run scripts with reading instructions inside
