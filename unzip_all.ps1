# PowerShell script to unzip all folders in the project
Write-Host "Starting to unzip all project folders..." -ForegroundColor Green

# Extract each zip file
if (Test-Path "static.zip") {
    Write-Host "Extracting static.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "static.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted static.zip" -ForegroundColor Green
}

if (Test-Path "templates.zip") {
    Write-Host "Extracting templates.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "templates.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted templates.zip" -ForegroundColor Green
}

if (Test-Path "uploads.zip") {
    Write-Host "Extracting uploads.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "uploads.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted uploads.zip" -ForegroundColor Green
}

if (Test-Path "pycache.zip") {
    Write-Host "Extracting pycache.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "pycache.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted pycache.zip" -ForegroundColor Green
}

if (Test-Path "Original_Dataset.zip") {
    Write-Host "Extracting Original_Dataset.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "Original_Dataset.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted Original_Dataset.zip" -ForegroundColor Green
}

if (Test-Path "Augmented_Dataset.zip") {
    Write-Host "Extracting Augmented_Dataset.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "Augmented_Dataset.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted Augmented_Dataset.zip" -ForegroundColor Green
}

if (Test-Path "plant_disease_env.zip") {
    Write-Host "Extracting plant_disease_env.zip..." -ForegroundColor Yellow
    Expand-Archive -Path "plant_disease_env.zip" -DestinationPath "." -Force
    Write-Host "Successfully extracted plant_disease_env.zip" -ForegroundColor Green
}

Write-Host "All folders have been extracted successfully!" -ForegroundColor Green
Write-Host "Extracted folders:" -ForegroundColor Cyan
Get-ChildItem -Directory | Where-Object { $_.Name -notlike "*.zip" } | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor White }

Write-Host "Cleaning up zip files..." -ForegroundColor Yellow
Get-ChildItem *.zip | Remove-Item -Force
Write-Host "Zip files cleaned up!" -ForegroundColor Green

Write-Host "Ready for deployment!" -ForegroundColor Green