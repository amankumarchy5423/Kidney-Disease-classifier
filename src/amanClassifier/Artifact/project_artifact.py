from dataclasses import dataclass



@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data : any
    test_data : any
    val_data : any

@dataclass(frozen=True)
class ModelBuildingArtifact:
    tuner : any


@dataclass(frozen=True)
class ModelTrainingArtifact:
    model_file : str
    preprocessor_file : str

# @dataclass(frozen=True)
# class ModelEvaluatorArtifact:
#     report : bool
#     bucket_name : str
#     ml_model_key : str
#     pre_model_key : str