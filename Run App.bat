@echo off
cd /d "%~dp0"
echo Starting the Gait Pattern Classifier...
echo The browser will open at http://localhost:8501 in a few seconds.
echo Keep this window open while using the app. Close it to stop the app.
echo.
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
python -m streamlit run app.py
echo.
echo The app has stopped.
pause