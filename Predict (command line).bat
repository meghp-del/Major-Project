@echo off
cd /d "%~dp0"
set /p F=Drag a .sto/.csv file or folder here and press Enter: 
python predict.py %F%
pause
