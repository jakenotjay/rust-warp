use thiserror::Error;

#[derive(Error, Debug)]
pub enum WarpError {
    #[error("Projection error: {0}")]
    Projection(#[from] ProjError),

    #[error("Resampling error: {0}")]
    Resampling(String),

    #[error("Invalid affine transform: {0}")]
    Affine(String),

    #[error("Invalid shape: {0}")]
    Shape(String),
}

#[derive(Error, Debug)]
pub enum ProjError {
    #[error("Unknown CRS: {0}")]
    UnknownCrs(String),

    #[error("Transform failed: {0}")]
    TransformFailed(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

#[derive(Error, Debug)]
pub enum PlanError {
    #[error("Planning error: {0}")]
    General(String),

    #[error("Projection error during planning: {0}")]
    Projection(#[from] ProjError),
}
