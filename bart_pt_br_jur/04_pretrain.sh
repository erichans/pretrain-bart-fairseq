source ../.venv/bin/activate
DATA_DIR=data-bin/tcu_jur_full/


fairseq-hydra-train -m --config-dir config/pretraining \
--config-name base task.data=$DATA_DIR 
#--checkpoint.restore_file=/path/to/roberta.base/model.pt
