import os

command = 'python run_classifier.py --data_dir /home/ubuntu/workplace/PTH/TextFooler-master/BERT/data/ag ' \
          '--bert_model roberta-base ' \
          '--task_name ag --output_dir results/ag --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)