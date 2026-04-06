param(
    [string] $model_name
)

Set-Location ..
omz_downloader --name $model_name
Set-Location .\scripts