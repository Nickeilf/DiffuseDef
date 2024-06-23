from .base import VanillaModel
from .diffuse import DiffuseModel, DiffuseModel2, DiffuseModel3
from .diffuse_rsmi import (
    DiffuseModelRSMI,
    DiffuseModelRSMIEnsembleAvg,
    DiffuseModelRSMI2,
    DiffuseModelRSMI2EnsembleAvg,
)
from .diffuse_ensemble import DiffuseModel2EnsembleAvg, DiffuseModel3EnsembleAvg

MODEL_CLASSES = {
    "vanilla": VanillaModel,
    "diffuse": DiffuseModel,
    "diffuse2": DiffuseModel2,
    "diffuse3": DiffuseModel3,
    "diffuse2rsmi": DiffuseModelRSMI,
    "diffuse2rsmi2": DiffuseModelRSMI2,
    "diffuse2rsmiensemavg": DiffuseModelRSMIEnsembleAvg,
    "diffuse2rsmi2ensemavg": DiffuseModelRSMI2EnsembleAvg,
    "diffuse2ensemavg": DiffuseModel2EnsembleAvg,
    "diffuse3ensemavg": DiffuseModel3EnsembleAvg,
}
