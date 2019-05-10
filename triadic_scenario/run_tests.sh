# Train on all train data
#CUDA_VISIBLE_DEVICES=1 nohup python -u pretrain_sibling_model.py 40 .00001 > outfiles/sibling/40_.00001.txt
#CUDA_VISIBLE_DEVICES=1 nohup python -u pretrain_sibling_model.py 40 .0001 > outfiles/sibling/40_.0001.txt
#CUDA_VISIBLE_DEVICES=1 nohup python -u pretrain_sibling_model.py 40 .001 > outfiles/sibling/40_.001.txt
CUDA_VISIBLE_DEVICES=1 nohup python -u sibling_and_child_model.py 40 .01  > outfiles/40_.01.txt

#CUDA_VISIBLE_DEVICES=1 nohup python -u one_child_model.py 40 .00001 > outfiles/50_40_.00001.txt
#CUDA_VISIBLE_DEVICES=1 nohup python -u one_child_model.py 40 .0001 > outfiles/50_40_.0001.txt
#CUDA_VISIBLE_DEVICES=1 nohup python -u one_child_model.py 40 .001 > outfiles/50_40_.001.txt
#CUDA_VISIBLE_DEVICES=1 nohup python -u one_child_model.py 40 .01  > outfiles/50_40_.01.txt
