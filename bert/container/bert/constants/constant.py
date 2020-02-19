import os,sys


hyperparameters = dict(epochs=10, lr=8e-5, max_seq_length=512, train_batch_size=16, lr_schedule="warmup_cosine",
                       warmup_steps=1000, optimizer_type="adamw")



training_config = dict(run_text="text-classification-bert", finetuned_model=None, do_lower_case="True",
                       train_file="train.csv", val_file="val.csv", label_file="labels.csv", text_col="message_body",
                       label_col='["abusive","asking_exchange","normal","offline_sell","possible_fraud","sharing_contact_details"]',
                       multi_label="True", grad_accumulation_steps="1", fp16_opt_level="O1", fp16="True",
                       model_type="roberta", model_name="roberta-base", logging_steps="300")