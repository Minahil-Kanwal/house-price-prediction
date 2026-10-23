<#!
.SYNOPSIS
Creates a local virtual environment, installs pinned dependencies, and trains the model.

.DESCRIPTION
This script is optional. It is a convenience wrapper for a clean Windows checkout;
the equivalent commands are documented in the README.
#>

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m src.cli train --download
