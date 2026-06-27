@echo off

echo fomating ..
black src tests notebooks

echo lint ..
flake8 src tests

echo tests .. 
pytest tests

echo Build predictor app image ..
docker build -t predictor-app:ci -f Dockerfile.predictor .
if errorlevel 1 exit /b 1

echo Run predictor app ..
docker run -d --name predictor-app -p 8000:8000 predictor-app:ci
if errorlevel 1 exit /b 1

echo Wait for app ..

set success=0

for /L %%i in (1,1,10) do (
    curl -f http://localhost:8000/health >nul 2>&1

    if not errorlevel 1 (
        set success=1
        goto :done
    )

    timeout /t 2 /nobreak >nul
)
:done

docker logs predictor-app
echo Stop container ..
docker rm -f predictor-app
echo Delete image ..
Docker rmi predictor-app:ci

if "%success%"=="1" (
    echo Health check passed.
    exit /b 0
)

echo Health check failed.
exit /b 1