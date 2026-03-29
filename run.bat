@echo off
REM Run the SDG pipeline via Blender headless mode.
REM
REM Usage: run.bat [--config config.yaml] [--start 0] [--end 50] [--debug]
REM
REM Set BLENDER env var if blender is not on your PATH:
REM   set BLENDER=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
REM   run.bat --config config.yaml

if "%BLENDER%"=="" set BLENDER=blender

"%BLENDER%" -b base_scene.blend -P run.py -- %*
