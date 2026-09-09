from dataclasses import dataclass



@dataclass(frozen=True)
class DataIngestionArtifact:
    data_ingestion_artifact : str

# @dataclass(frozen=True)
# class DataValidationArtifact:
#     data_dir : str

# @dataclass(frozen=True)
# class DataTransformationArtifact:
#     train_file : str
#     test_file : str
#     preprocessor_path : str

# @dataclass(frozen=True)
# class ModelTrainerArtifact:
#     model_file : str
#     preprocessor_file : str

# @dataclass(frozen=True)
# class ModelEvaluatorArtifact:
#     report : bool
#     bucket_name : str
#     ml_model_key : str
#     pre_model_key : str