DATA_DIR=/home/TCU/erichm/projetos/pretrain-bart-fairseq/roberta/data-bin/wikitext-103


fairseq-hydra-train -m --config-dir config/pretraining \
--config-name base task.data=$DATA_DIR 
#--checkpoint.restore_file=/path/to/roberta.base/model.pt
