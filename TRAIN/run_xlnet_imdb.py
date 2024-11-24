import os

command = 'python run_xlnet_classifier.py --data_dir /home/ubuntu/workplace/PTH/TextFooler-master/data/train/IMDB ' \
          '--bert_model xlnet-base-cased ' \
          '--output_dir /home/ubuntu/workplace/PTH/TextFooler-master/target_models/XLNET/imdb '\
          '--task_name imdb --do_eval --do_lower_case ' \
          '--train_batch_size 128 '\
          '--do_resume --do_train --do_lower_case'

os.system(command)