import os

command = 'python run_roberta_classifier.py --data_dir /home/ubuntu/workplace/PTH/TextFooler-master/data/train/MNLI' \
          '--bert_model roberta-base ' \
          '--output_dir /home/ubuntu/workplace/PTH/TextFooler-master/target_models/ROBERTA/MNLI '\
          '--task_name mnli --do_eval --do_lower_case ' \
          '--train_batch_size 128 '\
          '--do_resume --do_train --do_lower_case'

os.system(command)