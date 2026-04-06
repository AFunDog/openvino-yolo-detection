param(
    [string] $model_name
)

Set-Location ..
mo --input_model .\public\$model_name\$model_name.pb --transformations_config .\public\$model_name\$model_name.json --batch 1 --output_dir .\public\$model_name\ir_model
Set-Location .\scripts