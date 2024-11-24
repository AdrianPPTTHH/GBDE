import os

command = 'python run_xlnet_classifier.py --data_dir sst2 ' \
          '--bert_model xlnet-base-cased ' \
          '--output_dir /home/ubuntu/workplace/pth/TextFooler-master/target_models/XLNET/sst2 '\
          '--task_name sst2 --do_train --do_lower_case ' \
          '--train_batch_size 128 --max_seq_length 64 '\
          '--do_resume --do_train --do_lower_case'

os.system(command)