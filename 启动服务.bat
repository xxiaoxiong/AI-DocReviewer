@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 DocReviewer 启动中...
echo ============================================================
echo.

cd /d "%~dp0backend"

echo 📦 检查依赖...
python -c "import fastapi, uvicorn, docx, sklearn" 2>nul
if errorlevel 1 (
    echo ❌ 缺少依赖，正在安装...
    pip install python-docx scikit-learn python-multipart
)

echo.
echo ✅ 依赖检查完成
echo.
echo 🔑 配置 API Key...
set DEEPSEEK_API_KEY=sk-bc4ceb3384f244e38f596fd23631af63
echo    ✅ API Key 已配置
echo.
echo 🌐 启动服务...
echo    - API 文档: http://localhost:8000/docs
echo    - 前端页面: 请打开 frontend/index.html
echo.
echo ============================================================
echo.

python run.py

pause

