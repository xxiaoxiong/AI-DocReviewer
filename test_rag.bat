@echo off
chcp 65001 >nul
cd /d "%~dp0"
python test_rag_v2.py
pause

