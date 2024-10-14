import boto3
import json
import logging
import os
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
import sagemaker.session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.estimator import Estimator
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
import traceback



BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.

       Args:
           region: the aws region to start the session
           default_bucket: the bucket to use for storing the artifacts

       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
        region,
        sagemaker_project_arn=None,
        role=None,
        default_bucket=None,
        use_case=None,
        algorithm=None,
        tenant=None,
        model_package_group_name= None,
        pipeline_name=None,
        base_job_prefix=None,
        processing_instance_type="ml.t3.large",
        training_instance_type="ml.m5.large",
        inference_instance_type="ml.m5.large",
        ml_s3_bucket="xfactrs-devst-sagemaker",
        s3_prefix="NA",
        processing_instance_count_param= "1",
        training_instance_count_param= "1"
):
    print("tenant: ", tenant)
    pipeline_session = get_pipeline_session(region, default_bucket)

    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)

    training_hyperparameters = {
        "max_depth":9,
        "colsample_bytree":0.618753235646235,
        "gamma":0.20011144643601242,
        "learning_rate":0.1533144672640683,
        "subsample":0.901054624513367,
        "n_estimators":105,
        "reg_alpha":0.2641850606486046,
        "reg_lambda":0.218167915803278,
	    "num_round":1
    }
    
#    default_bucket = 'xfactrs-devst-sagemaker'
#    prefix = None

    input_data = ParameterString(
        name="InputData", default_value="s3://{}/{}/data/processing/input".format(ml_s3_bucket,s3_prefix)
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    processing_instance_count_param = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )

    training_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    
    
    
    process_image_uri = '738701072985.dkr.ecr.us-east-1.amazonaws.com/xfactrs-sagemaker-processing:1.21'
#    destination_train = Join(on="/",values=["s3:/",default_bucket,prefix,"data","training","train"])
#    destination_validation=Join(on="/",values=["s3:/",default_bucket,prefix,"data","training","validation"])
#    destination_test=Join(on="/",values=["s3:/",default_bucket,prefix,"data","test","input"])
    
    if s3_prefix != "NA": 
   
        input_data = f"s3://{ml_s3_bucket}/{s3_prefix}/{tenant}/{algorithm}/{use_case}/data/processing/input"
        destination_train = f"s3://{ml_s3_bucket}/{s3_prefix}/{tenant}/{algorithm}/{use_case}/data/training/train"
        destination_validation = f"s3://{ml_s3_bucket}/{s3_prefix}/{tenant}/{algorithm}/{use_case}/data/training/validation"
        destination_test = f"s3://{ml_s3_bucket}/{s3_prefix}/{tenant}/{algorithm}/{use_case}/data/test/input"
        model_output_path = f"s3://{ml_s3_bucket}/{s3_prefix}/{tenant}/{algorithm}/{use_case}/model"
    else:  
        input_data = f"s3://{ml_s3_bucket}/{tenant}/{algorithm}/{use_case}/data/processing/input"
        destination_train = f"s3://{ml_s3_bucket}/{tenant}/{algorithm}/{use_case}/data/training/train"
        destination_validation = f"s3://{ml_s3_bucket}/{tenant}/{algorithm}/{use_case}/data/training/validation"
        destination_test = f"s3://{ml_s3_bucket}/{tenant}/{algorithm}/{use_case}/data/test/input"
        model_output_path = f"s3://{ml_s3_bucket}/{tenant}/{algorithm}/{use_case}/model"
    
    # Print the constructed paths
    print("Input Data Path:", input_data)
    print("Destination Train Path:", destination_train)
    print("Destination Validation Path:", destination_validation)
    print("Destination Test Path:", destination_test)
    print("Model Output Path:", model_output_path)

    processor = ScriptProcessor(
                command=['python3'],
                image_uri=process_image_uri,
                role=role,
                instance_count=1,
                instance_type=processing_instance_type,
                sagemaker_session=pipeline_session)

       
    run_args = processor.get_run_args(
    inputs=[
        ProcessingInput(input_name="input_data",source=input_data, destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train",destination=destination_train),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation",destination=destination_validation),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/output/inference",destination=destination_test),
    ],
    code=os.path.join(BASE_DIR, "processing.py")
    )

    step_process = ProcessingStep(
        name="ProcessData",
        code=run_args.code,
        processor=processor,
        inputs=run_args.inputs,
        outputs=run_args.outputs
    )
    
    
    train_image_uri = '738701072985.dkr.ecr.us-east-1.amazonaws.com/xfactrs-sagemaker-training:1.21.0'
    # model_output_path = "s3://{}/{}/model".format(ml_s3_bucket,s3_prefix)
    
      
    estimator = Estimator(
         image_uri=train_image_uri,
         entry_point=os.path.join(BASE_DIR, "train.py"),
         framework_version="2.0.3",
         output_path=model_output_path,
         hyperparameters=training_hyperparameters,
         sagemaker_session=pipeline_session,
         role=role,
         instance_count=training_instance_count_param,
         instance_type=training_instance_type
     )
     
     
    step_train = TrainingStep(
         depends_on=[step_process],
         name="TrainModel",
         estimator=estimator,
         inputs={
             "train": TrainingInput(
                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                 content_type="text/csv"
                 ),
                 "validation": TrainingInput(
                     s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                     content_type="text/csv"
                     )
        }
    )
    

    step_register_model = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type]
    )


    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            processing_instance_count_param,
            training_instance_count_param
        ],
        steps=[
            step_process,
            step_train,
            step_register_model,
        ],
        sagemaker_session=pipeline_session
    )

    return pipeline
