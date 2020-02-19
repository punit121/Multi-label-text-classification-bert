#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sagemaker
from pathlib import Path
from sagemaker.predictor import json_serializer
from container.bert.constants.constant import *
import json


def train_deploy_chat_bert():

    role = sagemaker.get_execution_role()
    session = sagemaker.Session()

    # location for train.csv, val.csv and labels.csv
    DATA_PATH = Path("../data/")

    # Location for storing training_config.json
    CONFIG_PATH = DATA_PATH/'config'
    CONFIG_PATH.mkdir(exist_ok=True)

    # S3 bucket name
    bucket = 'sagemaker-deep-learning-bert-v2'

    # Prefix for S3 bucket for input and output
    prefix = 'sagemaker-deep-learning-bert-v2/input'
    prefix_output = 'sagemaker-deep-learning-bert-v2/output'


    with open(CONFIG_PATH/'training_config.json', 'w') as f:
        json.dump(training_config, f)



    # This is a  feature to upload data to S3 bucket

    s3_input = session.upload_data(DATA_PATH, bucket=bucket , key_prefix=prefix)

    session.upload_data(str(DATA_PATH/'labels.csv'), bucket=bucket , key_prefix=prefix)
    session.upload_data(str(DATA_PATH/'train.csv'), bucket=bucket , key_prefix=prefix)
    session.upload_data(str(DATA_PATH/'val.csv'), bucket=bucket , key_prefix=prefix)

    #  Creating an Estimator and start training

    account = session.boto_session.client('sts').get_caller_identity()['Account']
    region = session.boto_session.region_name

    image = "{}.dkr.ecr.{}.amazonaws.com/fluent-sagemaker-fast-bert:1.0-gpu-py36".format(account, region)

    output_path = "s3://{}/{}".format(bucket, prefix_output)

    estimator = sagemaker.estimator.Estimator(image,
                                              role,
                                              train_instance_count=1,
                                              train_instance_type='ml.p3.8xlarge',
                                              output_path=output_path,
                                              base_job_name='bert-text-classification-v1',
                                              hyperparameters=hyperparameters,
                                              sagemaker_session=session
                                             )

    estimator.fit(s3_input)


    # Deploy the model to hosting service


    predictor = estimator.deploy(1,
                                 'ml.m5.large',
                                 endpoint_name='bert-text-classification-v1',
                                 serializer=json_serializer)


if __name__=='__main__':
    train_deploy_chat_bert()
